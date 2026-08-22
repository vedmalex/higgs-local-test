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
from transformers import AutoModel, AutoTokenizer, GenerationConfig

if not hasattr(GenerationConfig, "generation_kwargs"):
    GenerationConfig.generation_kwargs = property(
        lambda self: self.__dict__.setdefault("generation_kwargs", {}),
        lambda self, val: self.__dict__.__setitem__("generation_kwargs", val),
    )

from stt_helper import MODEL_ID, REVISION, load_transcribe

REFERENCE = """Сегодня мы проверяем качество распознавания русской речи системой Higgs Audio.
Вриндаван находится в Индии. Шри Чайтанья Махапрабху учил повторению святого имени.
Кришна. Радхарани. Шримад-Бхагаватам. Гопала Бхатта Госвами. Радха-Раман."""


def peak_memory() -> dict:
    """Peak host RSS plus, on CUDA, peak device allocation for this process only."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kibibytes.
    peak = {"peak_host_rss_bytes": rss if platform.system() == "Darwin" else rss * 1024}
    if torch.cuda.is_available():
        peak["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
        peak["current_vram_bytes"] = torch.cuda.memory_allocated()
    return peak


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), required=True)
    parser.add_argument("--dtype", choices=("float16", "float32", "bfloat16"), required=True)
    parser.add_argument("--reference", type=Path, default=None,
                        help="Reference transcript for WER. Defaults to the built-in Russian fixture; "
                             "pass a file when the audio does not match it.")
    parser.add_argument("--metrics", type=Path, default=None, help="Write the metrics JSON to this file.")
    args = parser.parse_args()
    dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    diagnostics = {
        "model": MODEL_ID, "revision": REVISION, "device": args.device, "dtype": str(dtype),
        "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
        "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        diagnostics.update({
            "cuda_device": props.name,
            "cuda_capability": f"{props.major}.{props.minor}",
            "cuda_total_memory_bytes": props.total_memory,
            "cuda_bf16_supported": torch.cuda.is_bf16_supported(),
        })
        torch.cuda.reset_peak_memory_stats()
    print(json.dumps(diagnostics, indent=2))
    try:
        transcribe = load_transcribe()
        started = time.perf_counter()
        # On CUDA the shards stream straight into VRAM, so host RAM never holds a full
        # copy. On MPS the two-step CPU load stays deliberate: it keeps the failure
        # surface of the device move observable and separate from loading.
        placement = {"device_map": {"": 0}, "low_cpu_mem_usage": True} if args.device == "cuda" else {}
        model = AutoModel.from_pretrained(
            MODEL_ID, revision=REVISION, torch_dtype=dtype, trust_remote_code=True,
            attn_implementation="eager", **placement,
        )
        if args.device != "cuda":
            model.to(args.device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
        model.audio_out_bos_token_id = tokenizer.convert_tokens_to_ids("<|audio_out_bos|>")
        model.audio_eos_token_id = tokenizer.convert_tokens_to_ids("<|audio_eos|>")

        import functools
        model_cls = type(model)
        orig_sample = model_cls._sample
        orig_forward = model_cls.forward

        @functools.wraps(orig_sample)
        def _compat_sample(self, input_ids, logits_processor=None, stopping_criteria=None, generation_config=None, synced_gpus=False, streamer=None, past_key_values_buckets=None, **kwargs):
            return orig_sample(
                self,
                input_ids,
                logits_processor,
                stopping_criteria,
                generation_config,
                synced_gpus,
                streamer,
                past_key_values_buckets,
                **kwargs,
            )

        @functools.wraps(orig_forward)
        def _compat_forward(self, *args, **kwargs):
            kwargs.pop("tokenizer", None)
            return orig_forward(self, *args, **kwargs)

        model_cls._sample = _compat_sample
        model_cls.forward = _compat_forward
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
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(transcript + "\n", encoding="utf-8")
        reference = args.reference.read_text(encoding="utf-8").strip() if args.reference else REFERENCE
        diagnostics.update({
            "status": "PASSED", "model_load_seconds": load_seconds, "processing_seconds": processing,
            "audio_duration_seconds": duration, "rtf": processing / duration,
            "wer": wer(reference.lower(), transcript.lower()) if reference else None,
            "wer_reference": "builtin" if reference is REFERENCE else (str(args.reference) if reference else None),
            "transcript": transcript, "output": str(args.output),
            **peak_memory(),
        })
        if args.metrics:
            args.metrics.parent.mkdir(parents=True, exist_ok=True)
            args.metrics.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    except Exception as exc:
        failure = {**diagnostics, "status": "FAILED", "exception": repr(exc),
                   "traceback": traceback.format_exc(), **peak_memory()}
        if args.metrics:
            args.metrics.parent.mkdir(parents=True, exist_ok=True)
            args.metrics.write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        raise


if __name__ == "__main__":
    main()
