# M4-S0 — STT comparison (draft, no valid accuracy comparison yet)

Issue #57, M4 plan Track S, task S0 (`docs/research/audiobook/m4-plan.md` §4). Recorded
2026-08-25 on native Apple Silicon M1 (16 GB unified memory), same host and same fixture
(`samples/stt_ru.wav`, 60.00 s) as the existing Higgs STT and Qwen3-ASR 0.6B rows in `README.md`.

**This is a speed/memory comparison only.** No valid WER exists for any model below — see §2.
Do not read the table as an accuracy comparison.

## 1. What was run for this task

`mlx-community/Qwen3-ASR-1.7B-8bit` had been downloaded by `scripts/download_models.sh:67-68`
but never executed before this task. It was run once via the existing `--model` override in
`src/stt_qwen_local_test.py:70` (no code changes) against the same fixture used for the
already-recorded 0.6B and Higgs STT rows:

```bash
/usr/bin/time -l .venv-tts/bin/python src/stt_qwen_local_test.py \
  --model mlx-community/Qwen3-ASR-1.7B-8bit \
  --audio samples/stt_ru.wav \
  --output output/qwen17_stt_ru.txt \
  --metrics logs/qwen17_stt.json
```

Full output logged to `logs/qwen17_stt.log`; metrics JSON at `logs/qwen17_stt.json`.

### Transcript (Qwen3-ASR 1.7B-8bit, verbatim, untouched per AGENTS.md rule 6)

> Хари Кришна, дорогие вочна, примите, пожалуйста, мои поклоны. Поговорим сегодня о таком
> качестве брамана, как терпение. Очень важное, необходимое качество для того, чтобы развиваться
> в преднасосважении. Ома генад, тимиранасья, генанджина шалакая, чакшуритмилитамина,
> дасмаиши, гуравинама, ванчакалпата рупьяшча, крипасиндхупьявача, патита нампа ванипью, вочна и
> пиона мунама. Итак, терпение. Прочитаем. Для начала прочитаем стих из Бхагавад гиты, из
> второй главы, четырнадцатый стих. Атраспаршастукантия, шитошна сукхадухха, агаман панионитя,
> стам стучикшва.

Qualitative read (not a scored metric — no valid reference exists, see §2): compared to the
already-recorded 0.6B transcript in `logs/qwen_stt.log`, the 1.7B transcript gets the opening
Russian prose noticeably cleaner ("вочна" for "вайшнавы" is still wrong, but "преднасосважении"
vs 0.6B's "предновозжении" — both garble "преданном служении", neither is right) and both models
similarly mangle the embedded Sanskrit verse (`Bhagavad-gītā` 2.14). **This is an impression, not
a measurement** — without a matching reference transcript there is no scored WER to report for
either size.

### Timing and RTF

| Model | Load | Processing | Audio | RTF |
| ----- | ---: | ---------: | ----: | --: |
| Qwen3-ASR 0.6B-8bit (already recorded, `logs/qwen_stt.log`) | 3.14 s | 5.59 s | 60.00 s | 0.093 |
| Qwen3-ASR 1.7B-8bit (this task, `logs/qwen17_stt.log`) | 7.08 s | 17.46 s | 60.00 s | 0.291 |
| Higgs STT (already recorded, `logs/stt.log`, MPS/FP16) | 15.83 s | 28.56 s | 60.00 s | 0.476 |

**Honesty note on RTF:** the machine was under heavy, unrelated background load during this run
(`uptime` reported load average 52.98 immediately before the run — an unrelated Mahabharata
translation script was active in another session; per this task's instructions, no heavy
parallel process was started by this work itself). The pre-declared plan estimate for this model
class was RTF ≈0.086 (that number is actually the *0.6B* figure from `README.md`, carried into
the task brief as a rough expectation for "this class of model"); the measured 1.7B RTF of 0.291
is still far faster than Higgs STT's 0.476, but the absolute number should not be treated as a
clean-machine baseline. Re-running on an idle machine, in the style of M4-T1b, would be needed
for a load-free number.

### Memory — three explicitly named quantities per the plan's naming rule (§6, `m4-plan.md`)

| Model | `peak_mlx (GiB)` | `peak_footprint (GiB)` | `weights_on_disk (GiB)` |
| ----- | ---: | ---: | ---: |
| Qwen3-ASR 0.6B-8bit | 2.177 | not meaningful (see caveat) | 0.941 |
| Qwen3-ASR 1.7B-8bit | 3.577 | not meaningful (see caveat) | 2.298 |
| Higgs STT (FP16, MPS) | n/a — torch/MPS, no MLX allocator | 7.80 | 4.999 |

- `peak_mlx (GiB)` = `mx.get_peak_memory()`, read directly from `logs/qwen17_stt.json`
  (`peak_mlx_memory_bytes: 3841223532` = 3.577 GiB) and `logs/qwen_stt.log`
  (`peak_mlx_memory_bytes: 2337715276` = 2.177 GiB).
- `weights_on_disk (GiB)` = sum of the cached blob file sizes for each snapshot:
  2467859030 B (1.7B) = 2.298 GiB, 1010773761 B (0.6B) = 0.941 GiB,
  5367429273 B (Higgs STT) = 4.999 GiB.
- **`peak_footprint (GiB)` caveat, specific to `stt_qwen_local_test.py`'s process shape:** this
  script's runner process spawns a *child* process that does the actual model load/inference
  (AGENTS.md's process-isolation rule — "the runner never holds a model resident"). `/usr/bin/time
  -l` measures only the process it directly launches, i.e. the thin parent, not the child that
  does the real work. Both qwen runs show this directly: `logs/qwen17_stt.log`'s "peak memory
  footprint" line reads only 11060288 B (0.01 GiB) — far below even the child's own `ru_maxrss`
  of 827703296 B (0.771 GiB), which is itself the metric the plan says never to cite as *the*
  memory figure. **For this script's process shape, `/usr/bin/time -l`'s footprint number is not
  a valid whole-workload figure and is reported here as "not meaningful" rather than a number.**
  `peak_mlx (GiB)` is the correct ceiling to use for the Qwen rows. Higgs STT's `stt_test.py` does
  not spawn a child for the worker, so its `/usr/bin/time -l` footprint (7.80 GiB, already
  published in `README.md`) is a valid whole-process number for that row — but Higgs runs on
  torch/MPS, not MLX, so it has no `mx.get_peak_memory()` figure to report at all. The three-number
  scheme in `m4-plan.md` §6 was written with the MLX-based TTS profiler in mind and does not map
  cleanly onto a script that isolates model execution in a child process; this is recorded here so
  the gap doesn't get quietly papered over in a future rename (see `m4-plan.md` M4-T10).
- `ru_maxrss` (`resource.getrusage(...).ru_maxrss`, reported as `peak_host_rss_bytes` /
  `peak_memory_bytes` in these scripts' JSON) is **not cited** as a memory figure per the plan's
  rule — it undercounts MLX's unified-memory allocations on macOS.

## 2. Accuracy comparison — not possible yet

`README.md`'s previously published `WER 1.5` for Higgs STT has been retracted (this task, see the
README diff). Reading the fixture's `REFERENCE` text (`src/stt_test.py:24-26` — "Сегодня мы
проверяем качество распознавания русской речи системой Higgs Audio. Вриндаван находится в
Индии...") against the actual Higgs STT transcript (`logs/stt.log`, above) shows the two describe
entirely different content: the audio is a talk on the *quality of patience* (терпение) with a
Bhagavad-gītā citation, and the reference is a generic sentence about testing recognition quality
that never occurs in the recording. The Higgs transcript itself reads as fluent, plausible
Russian for what the audio actually contains — the 150%-error-rate number was never a measure of
model accuracy, only of the reference's mismatch to the file.

Qwen3-ASR 0.6B and 1.7B were run here with no `--reference` argument for exactly this reason —
both report `"wer": null` with `"wer_note": "no reference transcript supplied; WER not measured"`
(`logs/qwen_stt.log`, `logs/qwen17_stt.json`) rather than scoring against a reference known to be
wrong.

**No valid WER exists for Higgs STT, Qwen3-ASR 0.6B, or Qwen3-ASR 1.7B on this fixture.** A
verified reference transcript is a separate task (M4 plan Track S, S1 — "Build verified reference
transcripts once chapter-scale audio exists", planned for after Lane 2 produces real chapter
material). Until S1 lands, do not compute or cite a WER number for any model against
`samples/stt_ru.wav`.

## 3. Quantization caveat — not an architecture comparison

Both Qwen3-ASR checkpoints tested here (`mlx-community/Qwen3-ASR-0.6B-8bit` and
`-1.7B-8bit`) are **8-bit quantized**. Higgs STT runs in **FP16** on MPS. Any speed or memory gap
observed between the Qwen rows and the Higgs row in this document reflects some mixture of
architecture *and* quantization — the two variables are not separated here, and this project has
no FP16 Qwen3-ASR checkpoint and no 8-bit Higgs STT checkpoint to isolate them. Per the M4 plan's
explicit non-goal list (`m4-plan.md` §7): "**NOT** an architecture-level STT comparison — S2's
comparison is explicitly scoped to an 8-bit-vs-FP16 caveat, not a clean model-vs-model claim."
This document inherits that same scoping and is not exempt from it.

## 4. Whisper status — factual, no new work

`scripts/download_models.sh:55-56` downloads only `WhisperProcessor.from_pretrained(
"openai/whisper-large-v3")` — the tokenizer/feature-extractor, not a standalone Whisper model or
its encoder/decoder weights. There is **no standalone Whisper integration in this project.** The
actual Whisper-derived audio encoder weights this project uses live *inside* the
`bosonai/higgs-audio-v3-stt` checkpoint (Higgs STT's architecture is built on a Whisper encoder
per its model card) — they are not a separately loadable Whisper model. This means the owner's
"not only Whisper" requirement is already satisfied by the existing Higgs STT + Qwen3-ASR pair;
no standalone Whisper build was performed or is planned by this task, per `m4-plan.md` §7's
"NOT a standalone Whisper integration without a separate justification."

## 5. Summary table (speed/memory only — see §2 for why no accuracy column exists)

| Model | Precision | RTF | `peak_mlx (GiB)` | `peak_footprint (GiB)` | `weights_on_disk (GiB)` | WER |
| ----- | --------- | --: | ---: | ---: | ---: | --- |
| Qwen3-ASR 0.6B-8bit | 8-bit | 0.093 | 2.177 | not meaningful (child-process runner, see §1) | 0.941 | not valid — no reference (§2) |
| Qwen3-ASR 1.7B-8bit | 8-bit | 0.291 | 3.577 | not meaningful (child-process runner, see §1) | 2.298 | not valid — no reference (§2) |
| Higgs STT | FP16 | 0.476 | n/a (torch/MPS) | 7.80 | 4.999 | **retracted** — invalid fixture reference (§2) |

## 6. Status against S0's done condition

- 1.7B run: done, real numbers above, `logs/qwen17_stt.log` / `logs/qwen17_stt.json`.
- README `WER 1.5` retraction: done (`README.md`, one-line change, no new number invented).
- This is explicitly a draft comparison per the task brief — S1 (verified reference transcripts)
  and S2 (the actual accuracy comparison, still bound by the 8-bit-vs-FP16 caveat) remain open,
  unstarted, and are not claimed as complete here.
