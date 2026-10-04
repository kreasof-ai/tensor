# Producer schedule profiles

Profiles contain measured compiler settings outside the runtime package.
`tensor.compiler.search.ScheduleProfile` validates and selects entries;
`benchmarks/lfm2/producer.py --schedule-profile PATH` applies them through the
normal compiler interface. Generated bundles record the profile's canonical
SHA256 and each kernel's selected schedule.

The schema is `tensor.schedule-profile.v1`:

```json
{
  "schema": "tensor.schedule-profile.v1",
  "provider": "cuda",
  "target": "sm_86",
  "entries": [
    {
      "operation": "linear",
      "parameters": {"r": 1, "k": 2048, "o": 512, "type": 1},
      "schedule": {"threads": 256, "unroll": 4}
    }
  ]
}
```

Selectors may be partial. More specific selectors take precedence; conflicting
matches at equal specificity are errors. `runtime` optionally contains policy
consumed directly from the bundle, such as the grouped-attention transition.
`provenance` records the hardware, source revision and measurement evidence.

An explicit profile replaces the default profile. Missing entries use frontend
defaults. Revalidate and remeasure exported search winners in the full workload
before adopting them; the discovery harness's lowest observed time is not a
full-model performance result. Profiles are not portable performance claims.

The SM86 LFM2.5-2.6B profile initially preserves the settings measured at
`0cffba8`. See the [compiler cleanup report](../../../docs/research/lfm2-compiler-cleanup.md)
for frontend validation and any reselected schedules.
