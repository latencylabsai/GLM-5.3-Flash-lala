# Token speed

Median decode speed with native MTP and vision enabled, excluding prompt
processing. Both machines have 128 GB memory.

| Hardware | Code (tok/s) | Prose (tok/s) |
| --- | ---: | ---: |
| MacBook Pro, M5 Max | **52.23** | **28.88** |
| DGX Spark | **41.97** | **19.91** |

The shared [fixtures](../benchmarks/fixtures/sushi-4k.json) use 4,096 input
tokens, three greedy samples per workload, and 256 generated code tokens or
128 prose tokens. Both platforms use four MTP draft tokens, one active stream,
262,144-token context capacity, and FP16 attention KV. No input tokens were
cached. Throughput counts final output tokens and includes drafting and
verification time.

Raw results: [Mac](../benchmarks/records/mac-standalone-mtp-4k-mtp-2026-11-23.json)
and [DGX Spark](../benchmarks/records/spark-consistent-mtp-4k-mtp-2026-11-23.json).
See [validation](validation.md) for API, vision, and capacity checks.
