# M5 — Survey of open and commercial music-generation models

Issue #115. Recorded 2026-08-25. This is a **desk survey**, not a benchmark: nothing in this
document was installed, downloaded, or run on project hardware. Every claim below carries one of
three source markers, and they are load-bearing — do not strip them when reusing this material:

- **DOC** — verified against the model's own official documentation, README, or model card.
- **REVIEW** — taken from a third-party writeup, blog, or community report; not verified directly
  against a primary source by this project.
- **NOT FOUND** — actively searched for and not located (an absence of evidence, not evidence of
  absence).

The trigger for this survey was `<|env:music|>` / `<|env:noise|>` control-tag work
(`docs/research/*` control-tag lines, ids `151702`/`151703`) — a separate line of work tracked in
issue #116. This document answers a narrower, prior question: *which music-generation model, if
any, is worth trying on this project's M1/16 GB hardware, under what license, and with what
lyric/structure markup.*

## 1. Comparative table

| Model | Type | Vocals | Apple Silicon / MLX | Weight license | Commercial use | Structure markup |
|---|---|---|---|---|---|---|
| MusicGen (Meta) | instrumental + melody-conditioning | no | NOT FOUND official; third-party MLX port `musicgen-mlx` (MIT) | **CC-BY-NC 4.0** | FORBIDDEN | no tags; text + melody-audio conditioning |
| AudioGen (Meta) | sound effects | no | REVIEW: third-party blog reports MPS works | CC-BY-NC 4.0 | FORBIDDEN | none |
| Stable Audio Open | music/SFX, up to 47 s | no | REVIEW: unofficial MPS patch, upstream issue #79 | Stability AI Community License | up to $1M revenue/year | text prompt, `seconds_start`/`seconds_total` |
| Mustango | text + music-theory conditioning | no | NOT FOUND | MIT (code); weights presumed Apache-2.0 (**NOT verified**) | probably yes | claimed in the paper; no documented syntax found in the README |
| Riffusion (open) | spectrogram-as-image | no | **DOC**: MPS documented in README | CreativeML OpenRAIL-M | yes, with RAIL restrictions | none |
| YuE | full song | yes | NOT FOUND — CUDA only | Apache 2.0 | yes | `[verse]`/`[chorus]`/`[bridge]` (not `[intro]` — reported unstable) |
| ACE-Step / 1.5 | full song | yes | **DOC**: Mac/Apple-silicon official support + community MLX port; M2 Max ~26 s per minute of audio | Apache 2.0 (1.5 — possibly MIT, **REQUIRES VERIFICATION**) | yes | `[verse]/[chorus]/[bridge]/[intro]/[outro]/[instrumental]` |
| DiffRhythm / 2 | full song, fast | yes | **DOC**: "can now run on macOS", but not MLX | Apache 2.0 | yes | LRC timecodes (`[00:00.52]lyric`), no section tags |
| Magenta RealTime | real-time jam, instrumental | no (vocalizations only) | **DOC, strongest of all**: native Apple Silicon optimization, Small (230M) runs on a MacBook Air | code Apache 2.0, weights CC-BY 4.0 | yes | no lyrics; text/audio/MIDI conditioning |
| MusicLM (Google) | closed, unofficial reimplementations only | no | NOT FOUND | depends on the fork | n/a | effectively obsolete |

## 2. Markup detail per model

### YuE (DOC)

Lyric sections are separated by a blank line with a bracket label:

```
[verse]
...verse lyrics...

[chorus]
...chorus lyrics...
```

Plus a separate style line, e.g. `"inspiring female uplifting pop airy vocal electronic bright
vocal"`. Supported languages: English, Mandarin, Cantonese, Japanese, Korean. Generation proceeds
in ~30 s segments via `--run_n_segments`. No tempo or key control.

### ACE-Step

Tags: `[verse]/[chorus]/[bridge]/[intro]/[outro]/[instrumental]`. 19 languages; strongest coverage
in English/Chinese/Russian/Spanish/Japanese/German/French/Portuguese/Italian/Korean.

### DiffRhythm

A structurally different approach — per-line LRC timing rather than section tags. Structure is
implicit, not tagged:

```
[00:00.52]Abracadabra abracadabra
[00:03.97]Ha
```

### Magenta RealTime

Not a lyric model. Conditioning is a text prompt, an audio reference, or MIDI. No words will be
produced — vocalizations only.

### Mustango

Chord/beat/tempo/key control is claimed in the paper, but the README quickstart only shows
`model.generate(prompt)` with a plain text string — **no documented control syntax was found.**

## 3. Commercial APIs

| Service | Markup | Commercial terms | Risk |
|---|---|---|---|
| Suno | `[Verse]/[Chorus]/[Bridge]/[Hook]/[Drop]` (already covered by the `suno-music-producer` skill — see §6) | Free tier — non-commercial only; Pro/Premier — rights assignment, but the ToS explicitly states it "makes no representation ... that any copyright will vest in any Output" | High — unresolved Sony/UMG litigation |
| Udio | text + lyrics; `[Verse]`-style syntax not confirmed | Sources conflict; some reports place the June 2026 ToS at personal/non-commercial only on the free tier | Highest of the group |
| ElevenLabs Music | JSON composition plan: `sectionName`, `positiveLocalStyles`, `durationMs`, `lines` (up to 30 chunks) | Marketing claims broad commercial use on paid tiers plus training on licensed data; the primary license page returned a 404 when checked — **not verified** | Medium |
| Stability API | text prompt | Community License up to $1M revenue/year, Enterprise above that; Stable Audio 3.0 (May 2026) — WMG/UMG partnerships reported | Low |
| Google Lyria (Vertex AI) | text prompt | Paid API, commercial use permitted, **plus Google IP indemnification** | **Lowest of the group.** Do not confuse with the free MusicFX product, which is non-commercial and carries no indemnification |

**Legal background:** Sony/UMG v. Suno — the suit has expanded to cover roughly 61,000 recordings;
a summary-judgment hearing is scheduled for July 2026, with potential damages cited above $9
billion. Warner settled with Suno (November 2025). UMG settled with Udio (October 2025); Sony and
Warner have not settled with Udio as of this writing.

## 4. Applicability to the audiobook use case

- The project's actual need is **stings and background beds, not full songs with vocals** — this
  drops the hardware requirement by roughly an order of magnitude compared to the full-song models
  above.
- **Reproducibility**: generation is non-deterministic — the same prompt produces a different
  result each run. Practical consequence: **generate a sting once, save the WAV, and reuse it**,
  rather than regenerating on every book build.
- A full song with vocals on M1/16 GB is doubtful: YuE's official recommendation is 80 GB; ACE-Step
  is lighter (from ~8 GB), but the unofficial Mac port's ~26 s per minute of audio was measured on
  an M2 Max, and stability on an M1 with TTS memory already committed is not established.
- The connection to `<|env:music|>`/`<|env:noise|>` (ids 151702/151703) is a separate line of work,
  tracked in issue #116.

## 5. Recommendation — what to try first

1. **Stable Audio Open** — ~6 GB, community-verified MPS patch (~17 s vs. 51 s on CPU), license
   permits commercial use up to $1M revenue/year. Fits stings and background beds directly.
2. **ACE-Step-1.5** — Mac support is officially claimed, structural tags exist; start in
   instrumental (no-vocal) mode.
3. **Magenta RT (Small, 230M)** — worth trying if a live background music layer is ever needed
   under a scene; the only model with official Apple Silicon optimization.

**Do not start with YuE or DiffRhythm** on this hardware: YuE needs roughly an order of magnitude
more memory than is available; DiffRhythm has no confirmed MLX acceleration and no structural
tags.

## 6. Explicitly unverified — do not treat as settled

- The exact ACE-Step-1.5 weight license (Apache vs. MIT — sources disagree).
- The primary ElevenLabs Music license text (the page returned a 404 when checked).
- The current Udio ToS (conflicting reports on commercial-use terms).

## 7. Out of scope for this document

The `suno-music-producer` skill (`skills/suno-music-producer/SKILL.md` in the `agent-skills`
repository) already covers Suno production workflow in depth and was not touched or duplicated
here — §3's Suno row exists only for the comparative table, not as a substitute for that skill.

## References

- Issue #115 — this survey's tracking issue.
- Issue #116 — `<|env:music|>` / `<|env:noise|>` control-tag work (separate, related line).
- `docs/research/audiobook/m4-plan.md` — the broader M4 audiobook research track this survey
  supports.
