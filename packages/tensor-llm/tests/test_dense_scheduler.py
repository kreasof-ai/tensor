"""Scheduling contracts independent of CUDA or checkpoint availability."""

from tensor_llm.qwen35.dense.scheduler import Request, Scheduler


class FakeEngine:
    def __init__(self, capacity):
        self.capacity = capacity
        self.state = {}
        self.batches = []
        self.releases = []

    def admit(self):
        slot = next(i for i in range(self.capacity) if i not in self.state)
        self.state[slot] = 0
        return slot

    def release(self, slot):
        assert slot in self.state
        del self.state[slot]
        self.releases.append(slot)

    def batch(self, count, chunk):
        engine = self

        class Batch:
            def run(self, slots, sequences):
                engine.batches.append((len(slots), chunk))
                for slot, sequence in zip(slots, sequences):
                    engine.state[slot] += sum(sequence)
                return [engine.state[slot] % 100 for slot in slots]

        return Batch()

    def decode(self, slots, tokens):
        return self.batch(len(slots), 1).run(slots, [[t] for t in tokens])


def execute(requests, capacity):
    engine = FakeEngine(capacity)
    emitted = []
    scheduler = Scheduler(
        engine, emitted.extend, prefill_chunk=4, prefill_requests=min(4, capacity)
    )
    for request in requests:
        scheduler.submit(request)
    for _ in range(1000):
        if not scheduler.turn():
            break
    else:
        raise AssertionError("scheduler did not drain")
    return engine, scheduler, emitted


def test_compaction_and_continuous_refill_preserve_request_state():
    specifications = [([i + 1] * ((i % 3) + 1), 1 + (i % 7)) for i in range(32)]
    sequential = [Request(prompt, limit) for prompt, limit in specifications]
    batched = [Request(prompt, limit) for prompt, limit in specifications]
    _, _, _ = execute(sequential, 1)
    engine, scheduler, events = execute(batched, 8)
    assert [r.output for r in batched] == [r.output for r in sequential]
    assert not engine.state
    assert scheduler.stats["completed"] == 32
    assert scheduler.stats["max_active"] == 8
    assert min(map(int, scheduler.stats["capacity_counts"])) < 8
    assert (
        len([event for _, event in events if event["meta_info"]["finish_reason"]]) == 32
    )


def test_cancellation_releases_state_and_admits_queued_request():
    engine = FakeEngine(1)
    events = []
    scheduler = Scheduler(engine, events.extend, prefill_chunk=4)
    first = Request([1, 2, 3], 100)
    second = Request([8], 2)
    scheduler.submit(first)
    scheduler.submit(second)
    assert scheduler.turn()
    first.cancelled.set()
    while scheduler.turn():
        pass
    assert first.slot is None and second.slot is None and not engine.state
    assert second.output == [8, 16]
    assert scheduler.stats["cancelled"] == 1 and scheduler.stats["completed"] == 1


def test_unequal_prompts_and_lengths_drain_from_256_to_one():
    requests = [
        Request([i + 1] * ((i % 9) + 1), 1 if i < 255 else 20) for i in range(256)
    ]
    engine, scheduler, _ = execute(requests, 256)
    assert scheduler.stats["completed"] == 256 and not engine.state
    assert scheduler.stats["capacity_counts"]["1"] > 0
    assert max(count for count, chunk in engine.batches if chunk == 1) <= 256


class FakeSpeculativeEngine(FakeEngine):
    def decode_many(self, requests):
        sequences = []
        for request in requests:
            token = request.output[-1]
            sequence = []
            for _ in range(min(4, request.limit - len(request.output))):
                token = self.decode([request.slot], [token])[0]
                sequence.append(token)
            sequences.append(sequence)
        return sequences


def test_multiple_verified_tokens_preserve_limits_and_refill():
    specifications = [([i + 1] * ((i % 3) + 1), 1 + (i % 13)) for i in range(67)]
    control = [Request(prompt, limit) for prompt, limit in specifications]
    execute(control, 1)
    actual = [Request(prompt, limit) for prompt, limit in specifications]
    engine = FakeSpeculativeEngine(8)
    events = []
    scheduler = Scheduler(engine, events.extend, prefill_chunk=4, prefill_requests=4)
    for request in actual:
        scheduler.submit(request)
    while scheduler.turn():
        pass
    assert [r.output for r in actual] == [r.output for r in control]
    assert scheduler.stats["tokens"] == sum(limit for _, limit in specifications)
    assert scheduler.stats["completed"] == len(specifications) and not engine.state
    assert all(
        event["meta_info"]["completion_tokens"] == len(event["output_ids"])
        for _, event in events
    )
