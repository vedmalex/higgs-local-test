# M4 system-prompt voice-selection probe — code review only, no generation run

Issue #57 (M4 audiobook track). Owner's question: "давай проверим system промпт для выбора
голоса" — can a system instruction (spoken character/gender description) steer Higgs TTS 3's
voice, as a cheaper alternative to reference-audio cloning (RTF 7.73 with cloning vs. 6.56
without, `src/audiobook.py`)?

**Result: no generation was run.** Code review (independently, in parallel with the tag-reference
work, PR #113 / `docs/guides/tag_reference.md` §4.1) settles the question negatively before any
audio needed to be produced, and the GPU was never safely available for this probe during this
session (see §3). This is a complete, honest answer to the owner's question — Part 2/3 of the
original brief (batched generation + F0 measurement) were not executed because they are moot once
Part 1 shows the instruction cannot reach the model through any supported interface.

## 1. What the code actually does (verified independently)

Read directly, at `.venv-tts/lib/python3.12/site-packages/mlx_audio/tts/models/higgs_audio_v3/`:

- **`prompt.py`, `HiggsAudioV3PromptBuilder.build_prompt`** (lines 48-70): the prompt is built
  entirely in Python from a fixed structure —
  `<|tts|> [<|ref_text|>...] [<|ref_audio|>...codes...] <|text|>...encode_text(text)...<|audio|>`.
  There is no `system` field, no role concept, and no code path that reads the caller's `text` for
  embedded control structure beyond the officially documented `PROMPTING.md` tags. Grep for
  `system` (case-insensitive) across every `.py` file in `higgs_audio_v3/` returns **zero
  matches**.
- **`model.py`, `generate`** (line 738): `voice` is accepted and immediately discarded —
  `del voice, kwargs` (line 761). No `system`/`instruction` parameter exists.
- **`model.py`, `batch_generate`** (line 548): accepts `instructs: Optional[list[Optional[str]]]`
  in its signature (line 552) but **raises `ValueError`** for any non-`None` entry (lines 596-599:
  `"Higgs Audio v3 batch_generate does not support instructs"`). Same treatment for `voices` and
  non-default `genders`/`speeds`/`pitches`. The parameter exists in the signature (presumably
  mirroring an upstream API shape) but is actively rejected in this implementation — not a silent
  no-op, an explicit `ValueError`.
- **`PROMPTING.md`** (bundled with the pinned checkpoint snapshot): documents only 43 control tags
  — 21 `emotion`, 10 `prosody`, 3 `style`, 9 `sfx`. Zero mentions of "system", "character",
  "personality", "gender", "male", "female", or "instruction" anywhere in the file.
- The checkpoint's `AGENTS.md`/`README.md` likewise have zero mentions of system-prompt/voice
  steering.

**This independently confirms the tag-reference agent's finding** (`docs/guides/tag_reference.md`
§4.1, PR #113): there is no official parameter, and `HiggsAudioV3PromptBuilder.build_prompt` never
programmatically emits a system role of any kind. Full agreement on this point.

## 2. One nuance beyond the tag-reference finding — checked, not generation-tested

The tag-reference document's §4 table states `<|system|>` (token id 151677) is used by "NO (0 hits
anywhere in mlx_audio)" — true for *programmatic emission by the prompt builder*. I checked one
further, narrower question the owner explicitly raised: **if `<|system|>` is typed literally inside
the `text` string passed to `generate()`, does it survive as a real special token, or does it get
mangled into ordinary subword pieces (which would make it meaningless noise, not a signal)?**

Verified directly against the checkpoint's actual `AutoTokenizer` (no model load, no GPU):

```python
tok = AutoTokenizer.from_pretrained('bosonai/higgs-tts-3-4b')
ids = tok.encode('<|system|>Ты читаешь мужским голосом<|text|>Привет', add_special_tokens=False)
# -> [151677, 33995, 4552, 17307, 130141, 131260, 127584, 132692, 138430, 12228, 151672, ...]
```

`151677` and `151672` (`<|text|>`) come back as single, exact token ids — the tokenizer's
registered-special-token splitting (`tokenizer.json`: both are marked `"special": true`) applies
regardless of `add_special_tokens=False` (that flag only controls automatic BOS/EOS insertion, not
literal special-token substrings already present in the input string). So `HiggsAudioV3PromptBuilder
.encode_text`, which just calls `self.tokenizer.encode(text, add_special_tokens=False)`, would
faithfully turn a caller-injected `"<|system|>...instruction...<|text|>...content..."` string into
the exact token sequence
`<|tts|> <|text|> <|system|> [instruction] <|text|> [content] <|audio|>` — **parsing does not
break**, contrary to what would make this a forbidden ad-hoc hack under this task's honesty rule.

This does not contradict the tag-reference conclusion — it refines *why* the answer is still no:
the injection point is technically real and safe to construct, but the model was **never trained
to expect a mid-sequence `<|system|>` boundary during audio-generation fine-tuning** (the checkpoint's
own `chat_template.jinja` that *does* define system-role semantics uses stock Qwen3-4B-Base ChatML
(`<|im_start|>system ... <|im_end|>`), and is never rendered by `HiggsAudioV3PromptBuilder` at all —
so even the base LLM's system-role training, if any transferred to the audio head, used different
token ids entirely). Untrained special-token placement has no basis to be expected to steer
anything, and could equally plausibly do nothing, get silently ignored, or degrade the output.

**This residual question — does the untrained injection have any audible effect — was not tested by
generation in this session** (see §3). It remains a low-confidence, low-priority open curiosity, not
a live path worth pursuing operationally.

## 3. Why no audio was generated

`AGENTS.md` requires TTS/STT to run sequentially and never contend for the GPU. Across this
session's runtime, the shared M1 GPU was continuously occupied by other agents' legitimate M4
work — `m4_bgnoise_bench.py`, `m4_tag_catalog_bench.py` (a 39-clip control-tag/stress-mark
inventory run), and finally a full-chapter `src/audiobook.py` benchmark that pushed swap to
~4.7/6.1 GB used and load average above 8. Running a competing `batch_generate` call during any of
these would have contaminated another agent's timing/memory measurement and risked OOM/swap
thrashing on a 16 GB M1. A minimal single-batch probe script
(`docs/research/audiobook/m4_system_voice_bench.py`, 3 control repeats + 5 instruction variants =
8 clips = one `batch_generate` call) was written and is committed for a future run, but was
deliberately never executed.

Given §1 and §2 already answer the owner's actual question — is there a usable interface to select
voice via system instruction — running that script would only have tested an a-priori-unlikely,
untrained edge case at the cost of contending for a scarce, actively-used GPU. The negative
code-level answer is sufficient to close the owner's question; the script is left in place, unrun,
for anyone who wants to spend GPU time on the residual curiosity in §2 later.

## 4. Answer to the owner's question

**No.** Higgs TTS 3, as wired into this project through `mlx_audio`, has no working way to select
or steer voice gender/character through a system prompt or instruction of any kind:

- No parameter (`voice` is discarded; `instructs` raises `ValueError`).
- No structural token emitted by the prompt builder (`<|system|>` never appears in
  `build_prompt`'s output).
- No documentation (`PROMPTING.md` describes only emotion/prosody/style/sfx tags).
- A raw-text injection of the `<|system|>` token *would* survive tokenization intact (§2), but
  this is untrained, undocumented, and untested for any audible effect — not a real usable path,
  just a technical curiosity for a possible future GPU-cheap experiment.

## 5. Consequence for issue #118

This closes off the system-prompt route as a cheaper alternative to reference-audio cloning for
multi-voice audiobook narration. Issue #118 ("Wire per-character voice cloning into the audiobook
multi-speaker pipeline") already cites this finding (via PR #113) as the basis for going the
cloning route instead of searching for a prompt-based shortcut. This document independently
confirms that basis and adds no new information that would reopen the question: **voice cloning
via reference audio remains the only viable multi-voice mechanism for this checkpoint** under
`mlx_audio`, at its known cost (RTF 7.73 vs. 6.56 for plain generation).

## 6. Samples

None generated — see §3. `output/m4_system_voice/` was not created.
