# M4-T2 — Batching benchmark: results

Date: 2026-08-25. Refs #57. Hardware: Apple M1, 16 GB unified memory, macOS, native arm64,
`.venv-tts` (MLX-Audio, `bosonai/higgs-tts-3-4b`). Scope: `docs/research/audiobook/m4-plan.md` §3,
**M4-T2** ("Measure the existing batching implementation at batch sizes 2 and 4"). Batch size 8 is
also measured here: the plan's default guidance was to skip it (`≈11.9 GB of resident weights on a
16 GB machine leaves too little headroom for KV-cache growth`), but the machine was verifiably
freed up before this run (see §1), so batch 8 was measured too, per the instruction to climb until
memory actually binds rather than assume where it will. **It never did within this test's range —
see §3.**

This is a measurement of already-existing code (`continuous_batching.py`'s `BatchKVCache` /
`HiggsAudioV3BatchSession`, and the single-shot batched path `HiggsAudioV3.batch_generate` at
`model.py:548`). Nothing in `mlx_audio` was modified. Measurement script:
[`m4_batching_bench.py`](m4_batching_bench.py).

## 0. Method

- **API used**: `model.batch_generate(texts=[...], temperature=1.0, max_new_tokens=4096)`, an
  `Iterator[BatchGenerationResult]`. It left-pads prompts to the batch's longest prompt, runs one
  shared backbone forward per AR step across the whole batch (`BatchKVCache`), and evicts a row
  from the active batch as soon as that sequence's own `generation_done` fires — rows finish at
  different times but share compute until each exits. No equal-length requirement, no other
  functional restriction was hit; the sampler already independently tracks per-row stop state.
  It does reject `voices`, `instructs`, `speed`/`pitch`, and non-`"male"` `gender` per-item, all
  irrelevant here since audiobook sentiment travels as inline text tokens, not the `instruct` arg
  (§0.5 of the plan already established this; confirmed again empirically — no exception raised).
- **Workload**: 8 independent short Russian sentences (`SEGMENTS` in the script) — 6 of them are
  `samples/tts_ru.txt`'s own non-empty lines, reused verbatim for comparability with the project's
  other measured fixtures; 2 more were added in the same register/length only because batch size 8
  needs 8 distinct segments and the fixture has 6. This is deliberately "several independent short
  utterances," the shape a real audiobook actually has, not one long paragraph.
- **Design**: the *same* 8 segments are processed at every batch size, split into
  `ceil(8 / batch_size)` calls — 8 sequential `model.generate()` calls at batch 1 (today's
  production path = baseline), down to 1 `batch_generate()` call at batch 8. This makes the
  comparison apples-to-apples: identical text, identical total workload, only the batch depth
  changes.
- **Isolation**: one batch size per process, run to completion before the next was started, per
  project policy. Each run wrapped in `/usr/bin/time -l`; `peak_footprint` below is that wrapper's
  **`peak memory footprint`** line. **`maximum resident set size` is recorded in the raw logs but
  is never cited as a memory figure below** — this project has already confirmed it undercounts on
  macOS. `peak_mlx` is `mx.get_peak_memory()` read once at the end of each process, after
  `mx.reset_peak_memory()` was called right after the (discarded) warm-up generation, so it covers
  model weights + the actual batched KV-cache/activations for that run. `mx.eval()` is called
  explicitly on every produced audio array before any timer stops (see the script) — MLX is lazy,
  and the library's own `_decode_audio`/`batch_generate` code already does this internally too
  (`model.py:706, 719`), confirmed by reading it before relying on it.
- **Per-segment RTF is only meaningful at batch 1.** Inside one `batch_generate()` call, rows share
  every forward pass until each individually finishes (`continuous_batching.py:182-234`), so no
  wall-clock cost is attributable to one row in isolation — assigning the whole chunk's wall time to
  one segment's own duration would overstate its cost. The raw JSON logs carry `chunk_wall_seconds`
  per segment (which chunk it belonged to) with `rtf: null`; the table below reports the honest
  aggregate instead, which is also the actual metric the plan cares about (§2: "the success metric
  ... is how many batch slots now fit," and the payoff test is aggregate wall-time improvement).

## 1. Machine state (recorded before and after every run, not aggregated away)

| Run | Before: load avg (1/5/15) | Before: swap used/total | After: load avg (1/5/15) | After: swap used/total |
|---|---|---|---|---|
| batch=1 | 2.58 / 4.06 / 4.87 | 2.74 / 4.00 GB | 2.84 / 3.69 / 4.59 | 3.52 / 4.00 GB |
| batch=2 | 4.02 / 4.10 / 4.62 | 1.73 / 3.00 GB | 2.95 / 3.70 / 4.39 | 2.92 / 4.00 GB |
| batch=4 | 3.03 / 3.69 / 4.37 | 1.92 / 3.00 GB | 6.98 / 4.41 / 4.56 | 2.60 / 3.00 GB |
| batch=8 | 6.52 / 4.40 / 4.55 | 2.55 / 3.00 GB | 4.40 / 4.11 / 4.43 | 2.39 / 4.00 GB |

Context that matters for reading this table honestly: at the start of this task the machine was in
a genuinely bad state (load average ~6.5-6.8, swap 9.7 of 11.3 GB used, well past the load-52.98
mistake this project has already thrown out a measurement for once). The owner/coordinator closed
background load between the start of this task and these runs; by the time batch=1 started, swap
had shrunk to a 4 GB ceiling with under 3 GB used. Load average hovered 3-7 throughout (other
processes on a shared 10-user machine, not this benchmark alone) but swap never approached its
ceiling and no run showed symptoms of thrashing (no elapsed-time cliff, no error, no kill). This is
recorded so a future reader can judge these numbers' honesty for themselves rather than trust a
one-line assertion that "the machine was fine."

## 2. Results

| Batch size | Aggregate RTF | Throughput (audio-s / wall-s) | Speedup vs batch=1 | `peak_mlx` (GiB) | `peak_footprint` (GiB) | Wall time (8 segments) |
|---|---|---|---|---|---|---|
| 1 (baseline) | 3.882 | 0.2576 | 1.00x | 9.357 | 11.936 | 122.66 s |
| 2 | 3.416 | 0.2928 | 1.14x | 9.213 | 11.730 | 112.85 s |
| 4 | 2.031 | 0.4923 | 1.91x | 9.492 | 11.667 | 62.81 s |
| 8 | 1.142 | 0.8759 | **3.40x** | 9.161 | 11.778 | 28.50 s |

Per-segment RTF at batch=1 (the only batch size where it is separable): 3.857, 4.050, 3.895, 3.769,
3.971, 3.882, 3.873, 3.776 — mean 3.882, matching the aggregate exactly by construction (8
sequential single-segment calls). This baseline (RTF ≈ 3.9 on 8 short independent sentences) is in
the same impractical territory as the plan's own M4-T1 numbers (4.26 / 6.56 on a different, longer
fixture) — a different text, comparable order of magnitude, no contradiction.

Full raw JSON (per-run machine state, per-segment durations, `/usr/bin/time -l` output) is in
`logs/m4_batching_batch{1,2,4,8}.log` (this task's worktree) and reproduced in the PR.

## 3. Where does memory become the binding constraint?

**It did not, anywhere in this tested range (1, 2, 4, 8).** `peak_footprint` is flat within noise
across all four batch sizes — 11.94, 11.73, 11.67, 11.78 GiB — no upward trend with batch size at
all. This is the opposite of what the plan predicted going in (`batch 8 not tested: too little
headroom for KV-cache growth`).

The reason is visible in the numbers, not a surprise: `peak_footprint` is dominated by the ~11.4 GB
of resident model weights, which do not change with batch size. The `BatchKVCache` growth this
project's fixtures actually exercise is small: these are single short sentences (35-61 characters,
2.2-5.8 s of output audio, well under `max_new_tokens=4096`'s cap — the sampler's own
`generation_done` stops each row long before the cap). Eight rows' worth of KV cache for a few
hundred AR frames each is a rounding error next to the weight footprint. `peak_mlx` tells the same
story from the allocator's side: 9.16-9.49 GiB across all four runs, no trend.

**This is an important, explicitly flagged caveat, not a claim that batch 8 is unconditionally
safe:** a real audiobook has segments considerably longer than one short sentence (a paragraph, not
a clause), and KV-cache size scales with both batch depth *and* sequence length. This benchmark's
material was deliberately short (per the task's own instructions, matching how audiobook text is
actually chunked for TTS) and never grew the cache enough to find where it binds. **The batch-size
ceiling for realistic paragraph-length segments remains unmeasured** and is exactly the kind of
follow-up this document should not paper over. `mx.device_info()`'s
`max_recommended_working_set_size` on this M1 is ≈11.84 GiB (12,713,115,648 bytes) — every run
above already sits at or slightly past that *recommended* (not hard) ceiling, which is consistent
with why load/swap moved around during these runs even though nothing crashed.

## 4. Does batching clear the plan's threshold?

Plan §2: `batching payoff >= 1.5x wall-time improvement to be worth keeping`.

- Batch 2: **1.14x — does not clear the threshold on its own.**
- Batch 4: **1.91x — clears it.**
- Batch 8: **3.40x — clears it by a wide margin, and is the best configuration measured.**

**Verdict: batching is a real, large win, and the payoff keeps growing through the entire tested
range (1→2→4→8) without hitting the memory wall the plan expected.** Batch size 2 alone is not
worth adopting by itself; batch 4 already qualifies; batch 8 is unambiguously the best of the four
configurations measured and should be the starting point for anything downstream (T3/quantization,
Lane 2 chapter production), *for segments in this short-sentence length range* — see the caveat in
§3 before assuming this generalizes to longer paragraphs unchanged.

## 5. Ten-hour audiobook arithmetic, and suitability tier

Plan §2 tiers: `RTF <= 1.5 practical / 1.5-3 usable with mandatory resume support / > 3
impractical`.

| Config | RTF | 10-hour audiobook machine time | Tier |
|---|---|---|---|
| batch=1 (today) | 3.88 | ≈38.8 h | impractical |
| batch=2 | 3.42 | ≈34.2 h | impractical |
| batch=4 | 2.03 | ≈20.3 h | usable — mandatory resume support |
| **batch=8** | **1.14** | **≈11.4 h** | **practical** |

Batching alone — no quantization, no weight changes, nothing that could touch sentiment fidelity —
moves the best configuration from "impractical" all the way into the plan's "practical" tier
(RTF ≤ 1.5), *for this benchmark's short-sentence material*. This is a materially better outcome
than the plan anticipated going in (which expected batching to land in the 1.5-3 "usable" band at
best, with quantization needed afterward to reach "practical"). Whether this holds for
paragraph-length audiobook segments — where both the memory picture (§3) and the AR-loop
parallelism efficiency could differ — is the natural next check before treating "practical" as
settled for a real chapter.

## 6. What this does not answer

- **Longer, paragraph-length segments** (the actual unit `src/audiobook.py`'s segmentation
  produces for a real chapter) were not tested here; §3's memory-flatness result and §5's tier
  result are both conditioned on short sentences and may not transfer unchanged.
- **Batch sizes above 8** were not attempted; nothing in this run's evidence rules them out for
  short segments, but this was also not the task's scope.
- **Sentiment/control-tag fidelity under batching** is untouched by this document — `batch_generate`
  explicitly rejects the `instruct` argument, which does not matter for Higgs's inline text tokens
  (§0.5 of the plan), but no listening test was run here. That remains M4-T4's job, gated on M4-T3
  (quantization), not this measurement.
- **Quantization (M4-T3)** was not attempted; this document's result changes its priority (batching
  alone may already be sufficient for short-segment material) but does not make it moot for
  paragraph-length segments if §3's caveat turns out to matter.
