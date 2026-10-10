"""Exact greedy verification with protected state and accepted-prefix commit."""

from .mtp import DenseMTP


class SpeculativeEngine:
    def __init__(self, target, *, window=4, single_graph=False, deferred=False):
        if window not in (2, 4):
            raise ValueError("supported verification windows: 2, 4")
        self.target = target
        self.drafter = DenseMTP(target)
        self.window = window
        self.capacity = target.capacity
        self.context = target.context
        self.cached = {}
        self.stats = dict(rounds=0, request_rounds=0, proposed=0, accepted=0, emitted=0)
        self.single_graph = single_graph
        self.rounds = {}
        if deferred:
            if not single_graph:
                raise ValueError("deferred state requires a complete speculative graph")
            from .recurrent import RecurrentJournal

            target.deferred = RecurrentJournal(target, window)

    def round(self, count):
        from .spec_graph import SpeculativeRound

        slots = 1 << (count - 1).bit_length()
        if slots not in self.rounds:
            self.rounds[slots] = SpeculativeRound(self, count)
        return self.rounds[slots]

    def admit(self):
        slot = self.target.admit()
        self.drafter.admit_at(slot)
        return slot

    def release(self, slot):
        self.target.release(slot)
        self.drafter.release(slot)
        self.cached.pop(slot, None)

    def prefill_batch(self, requests, sequences, chunk):
        slots = [r.slot for r in requests]
        target = self.target.batch(len(slots), chunk)
        predicted = target.run(slots, sequences)
        shifted = []
        for request, sequence, token in zip(requests, sequences, predicted):
            end = request.offset + len(sequence)
            following = request.prompt[end] if end < len(request.prompt) else token
            shifted.append([*sequence[1:], following])
        proposals = self.drafter.batch(len(slots), chunk).run(
            slots, shifted, hidden=target.buffers["normal"]
        )
        self.cached.update(zip(slots, proposals))
        return predicted

    def decode_many(self, requests):
        if self.single_graph:
            return self.round(len(requests)).run(requests)
        slots = [r.slot for r in requests]
        remaining = [
            min(
                self.window,
                r.limit - len(r.output),
                self.context - int(self.target.positions[r.slot]),
            )
            for r in requests
        ]
        if max(remaining) == 1:
            return [
                [token]
                for token in self.target.decode(slots, [r.output[-1] for r in requests])
            ]
        proposals = [
            [self.cached[s]] if count > 1 else [] for s, count in zip(slots, remaining)
        ]
        before = [int(self.target.positions[s]) for s in slots]
        for depth in range(1, self.window - 1):
            active = [i for i, count in enumerate(remaining) if count > depth + 1]
            if not active:
                break
            predicted = self.drafter.batch(len(active)).run(
                [slots[i] for i in active], [[proposals[i][-1]] for i in active]
            )
            for i, token in zip(active, predicted):
                proposals[i].append(token)
        sequences = [
            [request.output[-1], *draft] for request, draft in zip(requests, proposals)
        ]
        verifier = self.target.batch(len(slots), self.window, verify=True)
        try:
            predictions = verifier.run(slots, sequences)
            accepted = []
            emitted = []
            for sequence, predicted in zip(sequences, predictions):
                count = 1
                for i, draft in enumerate(sequence[1:]):
                    if predicted[i] != draft:
                        break
                    count += 1
                accepted.append(count)
                emitted.append(predicted[:count])
            verifier.commit(accepted)
        except BaseException:
            if hasattr(verifier, "_verified"):
                verifier.discard()
            raise
        # Repair the draft cache with actual target hidden states and accepted
        # token IDs. Rejected speculative cache entries stay beyond its length.
        for slot, position in zip(slots, before):
            self.drafter.positions[slot] = position
        repaired = self.drafter.batch(len(slots), self.window).run(
            slots, emitted, hidden=verifier.buffers["normal"]
        )
        self.cached.update(zip(slots, repaired))
        self.stats["rounds"] += 1
        self.stats["proposed"] += sum(map(len, proposals))
        self.stats["request_rounds"] += len(slots)
        self.stats["accepted"] += sum(n - 1 for n in accepted)
        self.stats["emitted"] += sum(accepted)
        return emitted

    def close(self):
        for round in self.rounds.values():
            round.close()
        self.drafter.close()
        self.target.close()
