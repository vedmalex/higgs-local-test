#!/usr/bin/env python3
import argparse
import json
import platform
import resource
import time
import traceback
from pathlib import Path

import soundfile as sf
import torch
import transformers
from jiwer import wer
from transformers import AutoModel, AutoTokenizer

from stt_helper import MODEL_ID, REVISION, load_transcribe

REFERENCE = """Сегодня мы проверяем качество распознавания русской речи системой Higgs Audio.
Вриндаван находится в Индии. Шри Чайтанья Махапрабху учил повторению святого имени.
Кришна. Радхарани. Шримад-Бхагаватам. Гопала Бхатта Госвами. Радха-Раман."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), required=True)
    parser.add_argument("--dtype", choices=("float16", "float32"), required=True)
    args = parser.parse_args()
    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]
    diagnostics = {
        "model": MODEL_ID, "revision": REVISION, "device": args.device, "dtype": str(dtype),
        "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
        "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(),
    }
    print(json.dumps(diagnostics, indent=2))
    try:
        transcribe = load_transcribe()
        started = time.perf_counter()
        model = AutoModel.from_pretrained(
            MODEL_ID, revision=REVISION, torch_dtype=dtype, trust_remote_code=True,
            attn_implementation="eager",
        )
        model.to(args.device).eval()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        model.audio_out_bos_token_id = tokenizer.convert_tokens_to_ids("<|audio_out_bos|>")
        model.audio_eos_token_id = tokenizer.convert_tokens_to_ids("<|audio_eos|>")
        load_seconds = time.perf_counter() - started
        audio, sample_rate = sf.read(args.audio, dtype="float32")
        if sample_rate != 16000 or audio.ndim != 1:
            raise ValueError(f"expected mono 16 kHz audio, got sr={sample_rate}, shape={audio.shape}")
        duration = len(audio) / sample_rate
        started = time.perf_counter()
        transcript = transcribe(
            model, tokenizer, audio, sample_rate=sample_rate,
            user_prompt="Transcribe the Russian speech. Preserve Cyrillic. Output only the spoken words with no commentary.",
        )
        processing = time.perf_counter() - started
        args.output.write_text(transcript + "\n", encoding="utf-8")
        diagnostics.update({
            "model_load_seconds": load_seconds, "processing_seconds": processing,
            "audio_duration_seconds": duration, "rtf": processing / duration,
            "peak_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "wer": wer(REFERENCE.lower(), transcript.lower()), "transcript": transcript,
        })
        print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(json.dumps({**diagnostics, "exception": repr(exc), "traceback": traceback.format_exc()}, ensure_ascii=False, indent=2))
        raise


if __name__ == "__main__":
    main()
