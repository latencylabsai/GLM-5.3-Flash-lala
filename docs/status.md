# Release validation

Fresh installations on both platforms downloaded the public checkpoint without
Hugging Face authentication. All **27 files (99.95 GB)** passed hash verification,
and the download receipts match exactly.

Native MTP and vision are enabled, with 262,144-token context capacity and
16-bit KV storage.

| Check | MacBook Pro M5 Max, 128 GB | DGX Spark, 128 GB |
| --- | --- | --- |
| Runtime installation | Fresh Sushi build with verified release libraries | Fresh ExLlamaV3 build and pinned TabbyAPI |
| Model preparation | Fresh main and MTP packs; payload bytes verified | Native EXL3 checkpoint |
| Generated launch command | Passed | Passed |
| Chat, streaming, disconnect recovery | Passed | Passed |
| Red/blue image responses | Passed | Passed |
| 4K code/prose outputs | All six match saved serial reference text | All six match saved serial reference text |
| Long-context execution | 250,000 input + 32 output tokens; follow-up passed | 250,096 input + 32 output tokens; follow-up passed |

[Mac installation record](../benchmarks/records/mac-public-install-2026-11-24.json)
and [Spark installation record](../benchmarks/records/spark-public-install-2026-11-24.json)
contain the download hashes, runtime identities, and serving results.
The model revision is `29443247d589703b8e1e0244ba8242bda3aee077`.
See [token speed](benchmarks.md) for benchmark conditions and results.

Mac used Command Line Tools and downloaded Sushi libraries; no full Xcode or
MLX source build was needed. Spark used locally extracted official Ubuntu
Python development headers via `CPATH`; the documented header prerequisite
still applies.

Tests used a copied repository snapshot and empty model caches. Anonymous
GitHub cloning was not tested. Capacity checks use repeated public text and
do not establish retrieval quality. Red/blue image checks establish basic
image-conditioned serving, not general multimodal accuracy.
