"""Continuous admission and compact execution for request-owned model state."""

from collections import deque
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Event


@dataclass
class Request:
    prompt: list[int]
    limit: int
    channel: object = None
    cancelled: Event = field(default_factory=Event)
    slot: int | None = None
    offset: int = 0
    output: list[int] = field(default_factory=list)
    published: int = 0


class Scheduler:
    """Model-independent policy; engine implements admit/release/batch/decode.

    Stable cache slots are never execution row IDs. Prefill receives a bounded
    token budget each turn; decoding compacts its active requests every turn.
    """

    def __init__(self, engine, publish, *, prefill_chunk=32, prefill_requests=32):
        if prefill_chunk <= 0 or prefill_requests <= 0:
            raise ValueError("invalid prefill budget")
        self.engine = engine
        self.publish = publish
        self.prefill_chunk = prefill_chunk
        self.prefill_requests = prefill_requests
        self.incoming = Queue()
        self.pending = deque()
        self.prefilling = deque()
        self.decoding = []
        self.stopped = Event()
        self.stats = dict(
            tokens=0,
            completed=0,
            cancelled=0,
            decode_rounds=0,
            prefill_rounds=0,
            decode_rows=0,
            padded_decode_rows=0,
            max_active=0,
            capacity_counts={},
        )

    def submit(self, request):
        if self.stopped.is_set():
            raise RuntimeError("scheduler stopped")
        self.incoming.put(request)

    def _finish(self, request, *, cancelled=False):
        if request.slot is not None:
            self.engine.release(request.slot)
            request.slot = None
        self.stats["cancelled" if cancelled else "completed"] += 1

    def _emit(self, requests, tokens):
        return self._emit_sequences(requests, [[token] for token in tokens])

    def _emit_sequences(self, requests, sequences):
        if len(requests) != len(sequences):
            raise ValueError("missing generated request result")
        events = []
        continuing = []
        for request, tokens in zip(requests, sequences):
            if not tokens or len(tokens) > request.limit - len(request.output):
                raise ValueError("invalid generated token count")
            request.output.extend(tokens)
            self.stats["tokens"] += len(tokens)
            finished = len(request.output) >= request.limit
            # Publish the first token immediately. Later chunks amortize HTTP
            # overhead; they still contain actual newly generated token IDs.
            if (
                request.published == 0
                or len(request.output) - request.published >= 4
                or finished
            ):
                events.append(
                    (
                        request,
                        dict(
                            output_ids=list(request.output),
                            meta_info=dict(
                                prompt_tokens=len(request.prompt),
                                completion_tokens=len(request.output),
                                finish_reason=(
                                    dict(type="length", length=request.limit)
                                    if finished
                                    else None
                                ),
                            ),
                        ),
                    )
                )
                request.published = len(request.output)
            if finished:
                self._finish(request)
            else:
                continuing.append(request)
        if events:
            self.publish(events)
        return continuing

    def turn(self):
        while True:
            try:
                self.pending.append(self.incoming.get_nowait())
            except Empty:
                break
        self.decoding = [r for r in self.decoding if not self._cancel(r)]
        self.prefilling = deque(r for r in self.prefilling if not self._cancel(r))
        while (
            self.pending
            and len(self.decoding) + len(self.prefilling) < self.engine.capacity
        ):
            request = self.pending.popleft()
            if request.cancelled.is_set():
                continue
            request.slot = self.engine.admit()
            self.prefilling.append(request)
        self.stats["max_active"] = max(
            self.stats["max_active"], len(self.decoding) + len(self.prefilling)
        )
        if self.decoding:
            count = len(self.decoding)
            capacity = 1 << (count - 1).bit_length()
            self.stats["decode_rounds"] += 1
            self.stats["decode_rows"] += count
            self.stats["padded_decode_rows"] += capacity - count
            counts = self.stats["capacity_counts"]
            counts[str(capacity)] = counts.get(str(capacity), 0) + 1
            if hasattr(self.engine, "decode_many"):
                sequences = self.engine.decode_many(self.decoding)
                self.decoding = self._emit_sequences(self.decoding, sequences)
            else:
                tokens = self.engine.decode(
                    [r.slot for r in self.decoding],
                    [r.output[-1] for r in self.decoding],
                )
                self.decoding = self._emit(self.decoding, tokens)
        if self.prefilling:
            selected = [
                self.prefilling.popleft()
                for _ in range(min(len(self.prefilling), self.prefill_requests))
            ]
            sequences = [
                r.prompt[r.offset : r.offset + self.prefill_chunk] for r in selected
            ]
            if hasattr(self.engine, "prefill_batch"):
                tokens = self.engine.prefill_batch(
                    selected, sequences, self.prefill_chunk
                )
            else:
                batch = self.engine.batch(len(selected), self.prefill_chunk)
                tokens = batch.run([r.slot for r in selected], sequences)
            self.stats["prefill_rounds"] += 1
            finished = []
            first = []
            for request, sequence, token in zip(selected, sequences, tokens):
                request.offset += len(sequence)
                if request.offset == len(request.prompt):
                    finished.append(request)
                    first.append(token)
                else:
                    self.prefilling.append(request)
            self.decoding.extend(self._emit(finished, first))
        return bool(self.pending or self.prefilling or self.decoding)

    def _cancel(self, request):
        if not request.cancelled.is_set():
            return False
        self._finish(request, cancelled=True)
        return True

    def run(self):
        try:
            while not self.stopped.is_set():
                if not self.turn():
                    try:
                        self.pending.append(self.incoming.get(timeout=0.01))
                    except Empty:
                        pass
        finally:
            for request in (*self.prefilling, *self.decoding):
                if request.slot is not None:
                    self._finish(request, cancelled=True)
