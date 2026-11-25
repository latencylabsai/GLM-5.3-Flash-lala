# Matched Sushi 4K fixtures

`sushi-4k.json` contains the actual token-ID prompts captured from Sushi's
`bench --rungs 4k` requests, using the packaged GLM tokenizer and public Sushi
base `fc2fd843c2931f7cac150cdc86bb3873de273458` plus the packaged patch.

- Code: Sushi's deterministic synthetic TypeScript corpus, seed `0x5a5ac0de`.
- Prose: the opening of Jane Austen's *Pride and Prejudice*, public domain,
  from Sushi's bundled `src/fixtures/prose_pride_and_prejudice.txt`.
- Each prompt has exactly 4,096 tokens. Code requests 256 output tokens;
  prose requests 128, matching Sushi's default arm-specific lengths.

Credit to the Sushi contributors for the benchmark protocol and code corpus.
See [retained Sushi notices](../../patches/SUSHI-NOTICE) and
[license](../../patches/SUSHI-LICENSE).

The shared runner checks Spark's text decode/encode round trip against the
entire token array, not only its length. It runs three greedy samples per
arm with a 30-second pause between arms. All measured samples recomputed the
prompt (zero cached input tokens) in the tested serving profiles. This is a
matched-workload comparison, not a passing Sushi release-bench certification:
the latter requires prefix-cache restore behavior that this package disables.
