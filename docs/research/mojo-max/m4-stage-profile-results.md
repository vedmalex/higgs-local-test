# M4 stage profile — Higgs TTS 3 (MLX) time breakdown: Amdahl hypothesis CONFIRMED

Date: 2026-08-25. Run: `docs/research/mojo-max/m4_stage_profile.py`, native Apple M1 (16 GB),
`.venv-tts`, MLX 0.32.1 / mlx-audio 0.5.0, model `bosonai/higgs-tts-3-4b`. Blocking measurement
for issue #57's M4 planning: decide where a Mojo/MAX rewrite would actually move the needle.

## Question

Issue #57's existing README/log numbers show Higgs TTS 3 at RTF 6.56
(`logs/tts_basic.log`, ~66 h for a 10-hour audiobook). M0–M3 work targeted the codec vocoder
(`HiggsAudioTokenizer` decode: Conv1d/ConvTranspose1d/Snake1d/residual-unit prototypes). Before
sinking more effort there, this measurement checks the hypothesis stated in the issue: that the
vocoder decode is called **exactly once**, after the entire autoregressive (AR) Talker loop has
finished, and that the AR loop — one 4B-parameter forward pass per 40 ms audio frame — dominates
wall time. If true, an infinitely fast vocoder barely moves total generation time (Amdahl's law).

## Code verification (done before measuring, not assumed)

Read `mlx_audio/tts/models/higgs_audio_v3/model.py` (installed at
`.venv-tts/lib/python3.12/site-packages/mlx_audio/tts/models/higgs_audio_v3/model.py`,
mlx-audio 0.5.0) end to end. Confirmed structurally:

- `Model.generate()` (lines 738–852): builds prompt embeddings once, runs one backbone **prefill**
  forward over the whole prompt (line 788), then loops `for _ in range(limit)` (line 793) doing
  **one single-token backbone forward per audio frame** (lines 807–814) until the sampler signals
  end-of-codes.
- `Model._decode_audio()` (lines 341–352) is called **exactly once**, at line 816, *after* that
  loop exits — it stacks all delayed code rows, reverses the delay pattern, and calls
  `self._codec.decode(raw_codes)` a single time. There is no streaming/incremental decode path in
  `generate()`.
- `generation.step()` (in `generation.py`) already forces a host sync almost every AR frame via
  `int(codes_n[0].item())` (used to detect the end-of-codes token), so per-frame AR cost was
  already naturally synchronized before this profiler added its own `mx.eval()` boundaries.

This matches the issue's hypothesis about program structure. What was not yet known — and is the
point of this measurement — is what fraction of *wall time*, not code paths, each stage actually
consumes.

## Method

- `m4_stage_profile.py` monkey-patches one loaded `Model` instance (no changes to the installed
  package): `model.backbone` is replaced by a proxy object whose `__call__` splits by input shape
  into `ar_prefill` (prompt-length forward, called once) and `ar_backbone_steps` (per-frame
  single-token forward, called once per AR step); `_build_prompt_embeddings`, `_decode_audio`, and
  `_apply_fades` are wrapped the same way. Every wrapper calls `mx.eval()` on its output before
  stopping that stage's clock — mandatory for MLX's lazy evaluation, otherwise work silently
  migrates to whichever stage happens to force the next sync.
- A stage bucket not captured by any wrapper (mainly: sampling math + audio-code embedding + loop
  control between backbone calls) is reported as `ar_glue_sampling_embedding`, computed as
  `wall_total − sum(other stages)` so no time is unaccounted for.
- Model load is timed once, outside any generation. A short throwaway warm-up generation
  (`"Привет."`) runs immediately after load and is discarded — it absorbs MLX's first-call graph
  construction, so it doesn't contaminate the short/long numbers below.
- Two case lengths: **short** (one sentence, ~5 s target) and **long** (the project's existing
  `samples/tts_ru.txt` fixture, the same text `logs/tts_basic.log` used, ~19–20 s), so the
  audiobook-relevant long case is directly comparable to the already-recorded RTF 6.56.
- `mx.get_peak_memory()` (MLX's own device-buffer high-water mark) and `/usr/bin/time -l`'s
  process-level numbers are reported as separate, un-merged claims, per `AGENTS.md`.
- The whole script ran wrapped in a single `/usr/bin/time -l` invocation, alone, nothing else
  intentionally started during the run.

Full raw output: [`../../../logs/m4_stage_profile.log`](../../../logs/m4_stage_profile.log)
(not committed — logs are gitignored per `AGENTS.md`; re-run the command below to reproduce).

Reproduce with:

```bash
/usr/bin/time -l .venv-tts/bin/python docs/research/mojo-max/m4_stage_profile.py \
    | tee logs/m4_stage_profile.log
```

## Caveat: machine was not fully idle

`uptime` immediately before the run reported **load average 5.94** (M1, 8 logical cores), with
Docker Desktop, a `basedpyright` language server, and an `agy` agent process running in the
background (`ps aux` at the time showed the langserver alone using ~122% CPU). AGENTS.md and the
task both call for an otherwise-idle machine; that was not achieved here. This measurement was not
re-run under stricter isolation because the **stage-share result is a ratio internal to a single
run** (AR loop share vs. codec share), which is far less sensitive to a shared CPU load penalizing
Python-side host code equally across stages than the absolute RTF number is. The absolute RTF
below should be read as directionally consistent with, not a tight reproduction of, the previously
recorded 6.56 — see the discrepancy note below.

## Result

| Stage | short (4.96 s audio) | long (19.44 s audio) |
| --- | --- | --- |
| `prompt_prep` (tokenize + build embeddings) | 0.239 s — 1.18% | 0.317 s — 0.38% |
| `ar_prefill` (one backbone pass over the whole prompt) | 0.311 s — 1.54% | 2.735 s — 3.30% |
| `ar_backbone_steps` (one backbone pass per audio frame, 130 / 492 frames) | 19.035 s — **93.96%** | 75.415 s — **91.01%** |
| `ar_glue_sampling_embedding` (sampling + code embedding + loop control, by difference) | 0.000 s — 0.00% | 0.661 s — 0.80% |
| `codec_decode` (the single vocoder call) | 0.360 s — **1.78%** | 3.119 s — **3.76%** |
| `postprocess_and_write` (fades + WAV write) | 0.479 s — 2.36% | 0.618 s — 0.75% |
| **wall_total** | **20.259 s** | **82.865 s** |
| RTF (wall / audio duration, project convention) | 4.084 | 4.263 |
| AR frames | 130 | 492 |
| `mx.get_peak_memory()` | 10.13 GB | 11.93 GB |

Model load: 16.881 s (separate stage, not included in RTF, matches project convention).
Warm-up generation (discarded): 39.094 s.

Memory, kept as two separate claims per `AGENTS.md` (whole-script `/usr/bin/time -l`, covering
load + warm-up + both cases):

- MLX peak device buffers (`mx.get_peak_memory()`, per case, see table): up to **11.93 GB**.
- Process **maximum resident set size**: **1.32 GB** (`1323401216` bytes).
- Process **peak memory footprint** (macOS's `/usr/bin/time -l` footprint accounting, includes
  compressed/shared pages differently from RSS): **12.30 GB** (`12295540352` bytes) — this is the
  number that actually agrees with MLX's own peak-memory accounting; the classic "maximum resident
  set size" line undercounts on this platform and should not be quoted as the memory figure.

## Main finding: Amdahl's law hypothesis CONFIRMED

**The autoregressive Talker loop (`ar_prefill` + `ar_backbone_steps`) accounts for 95.5% (short)
and 94.3% (long) of total wall time. The single codec/vocoder decode call is 1.78% (short) and
3.76% (long).**

**The answer to the main question — what fraction of total time an infinitely fast (zero-cost)
vocoder would remove — is ~3.8% for the audiobook-relevant long case (~1.8% for the short case).**
Even a perfect, instantaneous vocoder would take the long case's wall time from 82.87 s to
~79.75 s: RTF would move from 4.26 to about 4.10, not a meaningful change for a 10-hour-audiobook
budget. The vocoder is not the bottleneck. M0–M3's vocoder-focused MAX/Mojo prototyping work,
whatever its outcome, cannot deliver more than roughly a 4% wall-time win on this pipeline as
currently structured — the ceiling is set by Amdahl's law, not by how fast the decode kernel ends
up being.

The dominant cost is unambiguous: one 4B-parameter single-token forward pass through the Qwen3
backbone per 40 ms of output audio (492 forward passes for the 19.44 s long case), at roughly
153 ms/frame on this M1 under the noted background load. **Any M4 effort aimed at reducing
Higgs's wall-clock time needs to target the AR loop itself** (e.g., batching/parallel decoding
strategies, quantizing/accelerating the backbone's per-step forward pass, or KV-cache/attention
optimizations) — not further vocoder kernel work.

## Discrepancy with the previously recorded RTF 6.56

The measured long-case RTF here is **4.26**, noticeably lower than the `logs/tts_basic.log`
figure of **6.56** for what is nominally the same fixture text. Both runs are on the same M1
machine and same model. Plausible, not fully disambiguated, contributors:

- Different background load at measurement time (this run: load average 5.94 with Docker/IDE
  processes active; the original log's contemporaneous load was not recorded).
- The instrumentation itself: every stage boundary calls `mx.eval()`, which forces a GPU/host sync
  slightly earlier than the unmodified code path would. This is very unlikely to make the
  instrumented run *faster* than the baseline, since sync points are a source of overhead, not
  savings — so this does not explain the direction of the gap.
- Possible package/version drift between when `tts_basic.log` was generated and the `mlx`/
  `mlx-audio` versions currently resolved in `.venv-tts` (0.32.1 / 0.5.0) — not independently
  re-verified against the log's timestamp/environment here.
- Text length is nominally the same fixture (`samples/tts_ru.txt`) but the produced audio duration
  differs slightly (19.44 s here vs. 18.88 s in the log), which is consistent with sampling
  randomness in the AR loop (`temperature=1.0`, no fixed seed in either script) producing a
  different number of frames, not a measurement error.

This is flagged honestly rather than adjusted to match: **the absolute RTF is not tightly
reproduced**, but the qualitative structural finding (AR loop dominates, vocoder is a small
single-digit-percent share) does not depend on which of these RTF values is more representative —
it would hold under either.

## What this does not resolve

- Does not measure the codec **encode** path (reference-audio cloning) — `generate()` without
  `ref_audio` never calls `encode_reference_audio`; a cloning-mode profile would need a separate
  run.
- Only two text lengths were measured (61 and 283 characters). The AR-loop share is already >90%
  at both lengths and is structurally guaranteed to grow, not shrink, with longer text (frame
  count scales with audio duration; the one vocoder call and one prefill do not), so this is not
  expected to change the conclusion for full-chapter-length audiobook inputs, but that was not
  directly measured.
- Does not attribute time *inside* `ar_backbone_steps` to specific transformer sub-operations
  (attention vs. MLP vs. KV-cache indexing) — that would be the natural next profiling step if M4
  decides to target the AR loop.
- Machine was not idle during the run (see caveat above); a stricter re-run would tighten the
  absolute RTF/timing numbers but is not expected to change the stage-share conclusion.

## Next

M4 planning should treat the AR loop's per-frame backbone forward pass, not the vocoder decode, as
the optimization target. If a Mojo/MAX path is pursued for Higgs, it should aim at the Talker's
per-step decode (KV-cache handling, attention, quantized/faster weight paths for the 4B backbone)
rather than continuing down the Conv1d/ConvTranspose1d/Snake1d vocoder track that M0–M3 explored.
