#!/usr/bin/env python3
"""Issue #57 M4 system-voice probe: does a `<|system|>...<|text|>...` prefix
change the generated voice's gender/character?

Context (established by code review before this run):
- `mlx_audio`'s `higgs_audio_v3` `generate`/`batch_generate` have no dedicated
  system-instruction parameter. `voice` is discarded (`del voice, kwargs`,
  model.py ~L761); `batch_generate`'s `instructs` kwarg exists in the
  signature but raises `ValueError` if not None (unsupported in this build).
- `HiggsAudioV3PromptBuilder.build_prompt` (prompt.py) only knows
  `<|tts|>`, `<|ref_audio|>`, `<|ref_text|>`, `<|text|>`, `<|audio|>` --
  no system role concept anywhere in the prompt-building code.
- The checkpoint's tokenizer DOES have a registered special token
  `<|system|>` (id 151677), and the checkpoint ships a generic ChatML
  `chat_template.jinja` (system/user/assistant roles via `<|im_start|>`)
  that `mlx_audio`'s TTS prompt builder never calls.
- Confirmed by direct tokenizer test: encoding a raw string containing the
  literal substring `<|system|>...<|text|>...` splits cleanly into the
  registered special-token ids (no parse break, no exception) exactly like
  a normal control tag. So this is a structurally safe (does-not-break-
  parsing) but functionally UNDOCUMENTED injection: nothing in PROMPTING.md
  or the bundled AGENTS.md/README.md describes system-prompt / character /
  gender steering for this checkpoint.

This script tests whether that injection has ANY audible effect, using
`model.batch_generate` (same API as `m4_tag_inventory_bench.py`,
issue #57 PR #105/#109 lineage).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from mlx_audio.audio_io import write as audio_write  # noqa: E402
from mlx_audio.tts.utils import load  # noqa: E402

MODEL_ID = "bosonai/higgs-tts-3-4b"
MAX_NEW_TOKENS = 4096
BATCH_SIZE = 8

S1 = "Сегодня я занимался повседневными делами."
S2 = "Утром я выпил чай и почитал книгу."
S3 = "Потом вышел на улицу и немного прошёлся."
NEUTRAL_TEXT = f"{S1} {S2} {S3}"

N_CONTROL_REPEATS = 3

SYSTEM_VARIANTS = [
    ("sys_ru_male", "Ты читаешь мужским низким голосом."),
    ("sys_ru_female", "Ты читаешь женским голосом."),
    ("sys_en_male", "Speak in a deep male voice."),
    ("sys_en_female", "Speak in a high female voice."),
    ("sys_ru_character", "Ты пожилой рассказчик, говоришь неспешно и веско."),
]


def build_clips() -> list[tuple[str, str]]:
    clips: list[tuple[str, str]] = []
    for i in range(1, N_CONTROL_REPEATS + 1):
        clips.append((f"control_{i}", NEUTRAL_TEXT))
    for clip_id, instruction in SYSTEM_VARIANTS:
        text = f"<|system|>{instruction}<|text|>{NEUTRAL_TEXT}"
        clips.append((clip_id, text))
    return clips


def machine_state() -> dict:
    def run(cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, check=False, capture_output=True, text=True).stdout.strip()
        except Exception as exc:  # pragma: no cover - diagnostic only
            return f"<error: {exc}>"

    return {"uptime": run(["uptime"]), "vm_swapusage": run(["sysctl", "vm.swapusage"])}


def audio_duration(samples: int, sample_rate: int) -> float:
    return samples / sample_rate if sample_rate else 0.0


def main() -> None:
    out_dir = ROOT / "output" / "m4_system_voice"
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = build_clips()
    print(f"total clips: {len(clips)}", flush=True)
    for clip_id, text in clips:
        print(f"  {clip_id}: {text!r}", flush=True)

    state_before = machine_state()
    print("machine state before run:", json.dumps(state_before, ensure_ascii=False), flush=True)

    load_start = time.perf_counter()
    model = load(MODEL_ID, model_type="higgs_audio_v3")
    load_seconds = time.perf_counter() - load_start
    print(f"model_load_seconds={load_seconds:.3f}", flush=True)

    t0 = time.perf_counter()
    list(model.generate(text="Это короткая прогревочная фраза перед замером.",
                         temperature=1.0, max_new_tokens=MAX_NEW_TOKENS))
    warmup_seconds = time.perf_counter() - t0
    print(f"warmup_seconds={warmup_seconds:.3f} (discarded)", flush=True)

    mx.reset_peak_memory()
    run_start = time.perf_counter()

    per_clip = []
    manifest = []
    for chunk_start in range(0, len(clips), BATCH_SIZE):
        chunk = clips[chunk_start: chunk_start + BATCH_SIZE]
        texts = [t for _, t in chunk]
        t0 = time.perf_counter()
        results = list(
            model.batch_generate(texts=texts, temperature=1.0, max_new_tokens=MAX_NEW_TOKENS)
        )
        mx.eval(*[r.audio for r in results])
        chunk_wall = time.perf_counter() - t0
        results.sort(key=lambda r: r.sequence_idx)
        if len(results) != len(chunk):
            raise RuntimeError(
                f"chunk starting at {chunk_start}: expected {len(chunk)} results, got {len(results)}"
            )
        for offset, result in enumerate(results):
            clip_id, text = chunk[offset]
            audio = np.asarray(result.audio).reshape(-1)
            sample_rate = result.sample_rate
            duration = audio_duration(len(audio), sample_rate)
            wav_path = out_dir / f"{clip_id}.wav"
            audio_write(str(wav_path), audio, sample_rate)
            per_clip.append(
                {
                    "clip_id": clip_id,
                    "text": text,
                    "chars": len(text),
                    "chunk_wall_seconds": chunk_wall,
                    "chunk_size": len(chunk),
                    "audio_duration_seconds": duration,
                    "wav_path": str(wav_path),
                }
            )
            manifest.append(clip_id)
        print(f"  [batch={len(chunk)}] {manifest[-len(chunk):]}: wall={chunk_wall:.3f}s", flush=True)

    run_wall = time.perf_counter() - run_start
    total_audio = sum(c["audio_duration_seconds"] for c in per_clip)
    aggregate_rtf = (run_wall / total_audio) if total_audio else None
    peak_mlx_gib = mx.get_peak_memory() / (1024 ** 3)

    state_after = machine_state()

    result = {
        "num_clips": len(clips),
        "batch_size": BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "model_load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "run_wall_seconds": run_wall,
        "total_audio_duration_seconds": total_audio,
        "aggregate_rtf": aggregate_rtf,
        "peak_mlx_gib": peak_mlx_gib,
        "per_clip": per_clip,
        "machine_state_before": state_before,
        "machine_state_after": state_after,
    }
    out_json = ROOT / "logs" / "m4_system_voice.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
