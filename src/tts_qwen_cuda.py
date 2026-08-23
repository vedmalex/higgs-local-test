#!/usr/bin/env python3
"""CUDA TTS runner for Qwen3-TTS, a second independent TTS backend (#52).

Qwen3-TTS is added as a diagnostic control on the same Colab T4 that reproducibly
fails to produce real speech with Higgs (#48), and as a second audiobook-production
candidate. It never replaces Higgs: this runner is a separate script producing a
separate metrics file, and its results must never be reported as a Higgs PASS.

Served the same way this project already serves Higgs — `vllm serve --omni` from
vllm-omni, exposing `POST /v1/audio/speech` — but with one server per model
*variant* (Base / CustomVoice / VoiceDesign), because a request's `task_type` only
works against the checkpoint that was trained for it: a CustomVoice server cannot
serve a voice-clone request, and vice versa. See
`docs/research/qwen3-tts-notes.md` for the source material behind every choice
below; nothing here is invented past what that research recorded.

This runner never fabricates a result: an unmet requirement is reported as SKIPPED
with the reason, a mismatched task_type/variant combination is reported as SKIPPED
rather than silently attempted, and a failure keeps its traceback.
"""
import argparse
import base64
import json
import os
import platform
import shutil
import subprocess
import time
import traceback
from pathlib import Path

from tts_cuda_common import (
    BF16_CAPABILITY,
    DeviceMemorySampler,
    gpu_facts,
    peak_memory,
    synthesize,
    terminate_server,
    wait_for_server,
    wav_duration,
)

# Confirmed exact Hugging Face IDs (docs/research/qwen3-tts-notes.md). There is no
# 0.6B VoiceDesign checkpoint, so Phase 1 (T4 diagnostic) covers Base + CustomVoice
# only; VoiceDesign is Phase 2 (1.7B), as the issue specifies.
MODEL_VARIANTS = {
    "0.6b-base": {
        "model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "basic": False, "clone": True, "style": False, "voicedesign": False,
    },
    "0.6b-customvoice": {
        "model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "basic": True, "clone": False, "style": True, "voicedesign": False,
    },
    "1.7b-base": {
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "basic": False, "clone": True, "style": False, "voicedesign": False,
    },
    "1.7b-customvoice": {
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "basic": True, "clone": False, "style": True, "voicedesign": False,
    },
    "1.7b-voicedesign": {
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "basic": False, "clone": False, "style": True, "voicedesign": True,
    },
}
DEFAULT_VARIANT = "0.6b-base"  # Phase 1 T4 diagnostic model named in the issue.

# vllm-omni's Qwen3-TTS deploy profile does not pin an attention backend (unlike
# Higgs's, which pins FLASHINFER and is exactly what aborts startup below compute
# 8.0 -- see configs/higgs_multimodal_qwen3_turing.yaml). So no Turing override is
# shipped here yet: docs/research/qwen3-tts-notes.md records this as unconfirmed
# until measured on a real T4, not as a settled fact. If a real run hits an
# analogous attention-backend or dtype-cast failure, add
# configs/qwen3_tts_turing.yaml then -- do not pre-author a speculative override.
TURING_DEPLOY_CONFIG = (Path(__file__).resolve().parents[1]
                        / "configs/qwen3_tts_turing.yaml")

# vLLM-Omni's speech endpoint historically caps voice-reference length for Higgs;
# whether the same limit applies to Qwen3-TTS's Base task_type is unconfirmed. The
# runner does not invent a cap here -- the server is the authority on this -- but
# still records a reference that is implausibly long for docs/guides/voice_cloning_guide.md's
# 7-12s recommendation so a SKIPPED reason stays informative rather than silent.
LONG_REFERENCE_WARNING_SECONDS = 30.0

DEFAULT_VOICE = "vivian"  # One of the 9 predefined CustomVoice timbres.

# Never an invented tag DSL (the issue explicitly forbids that) -- a plain
# natural-language instruction, matching the issue's own example intent, in the
# `instructions=` field the CustomVoice/VoiceDesign task_type actually documents.
STYLE_INSTRUCTION = (
    "Читай медленно и вдумчиво, тёплым, задумчивым тоном. Веди повествование "
    "сдержанно, без театральности, и стань немного более эмоциональным к концу "
    "абзаца."
)
VOICEDESIGN_INSTRUCTION = (
    "Создай голос зрелого рассказчика для аудиокниги: тёплый, спокойный мужской "
    "голос среднего возраста, с чёткой дикцией и мягким, доверительным тоном, без "
    "театральности."
)


def check_requirements(min_capability: tuple[int, int] | None) -> tuple[bool, str, dict]:
    """Return (runnable, reason, facts). Only genuinely missing prerequisites block."""
    facts = gpu_facts()
    if facts.get("gate") == "torch_missing":
        return False, (
            "torch is missing from this environment, which means the vLLM-Omni "
            "install never completed here"
        ), facts
    if facts.get("gate") == "no_cuda":
        return False, "no CUDA device is visible to torch", facts
    if shutil.which("vllm") is None:
        return False, "the `vllm` CLI is not installed, so this backend cannot run", facts

    capability = facts["cuda_capability_tuple"]
    if min_capability is not None and capability < min_capability:
        return False, (
            f"{facts['cuda_device']} has compute capability {facts['cuda_capability']}, "
            f"below the --min-capability {'.'.join(map(str, min_capability))} requested "
            "for this run"
        ), facts
    return True, "", facts


def server_command(model_dir: str, args, capability: tuple) -> list:
    command = ["vllm", "serve", model_dir, "--trust-remote-code", "--omni",
               "--host", "127.0.0.1", "--port", str(args.port)]
    if capability < BF16_CAPABILITY:
        # The checkpoint declares bfloat16 (docs/research/qwen3-tts-notes.md); vLLM
        # refuses it below compute 8.0, same gate already measured for Higgs.
        command += ["--dtype", "float16"]
        if args.deploy_config is None and TURING_DEPLOY_CONFIG.exists():
            command += ["--deploy-config", str(TURING_DEPLOY_CONFIG)]
    if args.deploy_config is not None:
        command += ["--deploy-config", str(args.deploy_config)]
    if args.mem_fraction_static is not None:
        command += ["--gpu-memory-utilization", str(args.mem_fraction_static)]
    return command


def attention_backend_diagnostics(log_path: Path) -> dict:
    """Record which attention backend vLLM actually selected for this run.

    Unlike Higgs, Qwen3-TTS's upstream deploy profile does not pin one -- so which
    backend vLLM's auto-selector picks on a T4 is an open question this run answers,
    not an assumption (docs/research/qwen3-tts-notes.md).

    Matches vLLM's own "Using X attention backend" log line rather than a bare
    substring search: a bare search for "FLASHINFER" false-positived on the
    unrelated `VLLM_USE_FLASHINFER_SAMPLER` env-var name mentioned in a warning
    line, reporting FLASHINFER even on a run where TRITON_ATTN was the backend
    actually selected (observed on a real T4 run, #52).
    """
    import re

    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Using (\w+) attention backend", text)
    if matches:
        return {"attention_backend_observed": list(dict.fromkeys(matches))}
    return {"attention_backend_observed": None,
            "attention_backend_note": "no 'Using X attention backend' line found in the server log"}


def build_jobs(args, variant: dict, model_dir: str) -> list[dict]:
    """One job per capability the loaded model variant actually supports.

    A job whose task_type the loaded variant does not support is SKIPPED with the
    variant name in the reason, rather than attempted and misreported.
    """
    jobs = []
    common = {"response_format": "wav", "max_new_tokens": args.max_new_tokens,
              "language": args.language}

    if variant["basic"]:
        if args.text_file and args.text_file.exists():
            basic_text = args.text_file.read_text(encoding="utf-8").strip()
        else:
            basic_text = ""
        if basic_text:
            jobs.append({
                "name": "qwen_tts_basic",
                "payload": {"input": basic_text, "task_type": "CustomVoice",
                            "voice": args.voice, **common},
                "output": args.output_dir / "qwen_tts_ru_basic.wav",
            })
        else:
            jobs.append({"name": "qwen_tts_basic",
                        "skipped": f"no synthesis text: {args.text_file} is missing or empty"})
    else:
        jobs.append({"name": "qwen_tts_basic",
                    "skipped": (f"basic Russian TTS needs a CustomVoice model; "
                                f"this run loaded {args.model_variant}")})

    if variant["clone"]:
        reference_seconds = None
        if args.ref_audio and args.ref_audio.exists():
            try:
                reference_seconds = wav_duration(args.ref_audio)
            except Exception:
                reference_seconds = None
        if not (args.ref_audio and args.ref_text
                and args.ref_audio.exists() and args.ref_text.exists()):
            jobs.append({"name": "qwen_tts_clone",
                        "skipped": "voice cloning needs both reference audio and its exact transcript"})
        else:
            if reference_seconds and reference_seconds > LONG_REFERENCE_WARNING_SECONDS:
                print(f"⚠️  reference audio is {reference_seconds:.1f}s; "
                      f"docs/guides/voice_cloning_guide.md recommends 7-12s. "
                      "Attempting anyway -- the server is the authority on any limit.")
            reference_text = args.ref_text.read_text(encoding="utf-8").strip()
            clone_text = (args.text_file.read_text(encoding="utf-8").strip()
                          if args.text_file and args.text_file.exists() else STYLE_INSTRUCTION)
            # docs/research/qwen3-tts-notes.md did not pin the exact wire format for
            # `ref_audio` on Qwen3-TTS's Base task_type; this reuses the vLLM-Omni
            # `data:` base64 URL format already confirmed for Higgs on the same
            # speech API (src/tts_cuda.py's reference_payload for backend="vllm"),
            # since both are served by the same vLLM-Omni /v1/audio/speech endpoint.
            mime = "audio/wav" if args.ref_audio.suffix.lower() == ".wav" else "audio/mpeg"
            encoded = base64.b64encode(args.ref_audio.read_bytes()).decode("ascii")
            jobs.append({
                "name": "qwen_tts_clone",
                "payload": {"input": clone_text, "task_type": "Base",
                            "ref_audio": f"data:{mime};base64,{encoded}",
                            "ref_text": reference_text, **common},
                "output": args.output_dir / "qwen_tts_ru_clone.wav",
            })
    else:
        jobs.append({"name": "qwen_tts_clone",
                    "skipped": f"voice cloning needs a Base model; this run loaded {args.model_variant}"})

    if variant["style"] and not variant["voicedesign"]:
        jobs.append({
            "name": "qwen_tts_style",
            "payload": {"input": args.text_file.read_text(encoding="utf-8").strip()
                        if args.text_file and args.text_file.exists() else STYLE_INSTRUCTION,
                        "task_type": "CustomVoice", "voice": args.voice,
                        "instructions": STYLE_INSTRUCTION, **common},
            "output": args.output_dir / "qwen_tts_ru_style.wav",
        })
    elif not variant["voicedesign"]:
        jobs.append({"name": "qwen_tts_style",
                    "skipped": (f"emotion/style instruction test needs a CustomVoice model; "
                                f"this run loaded {args.model_variant}")})

    if variant["voicedesign"]:
        jobs.append({
            "name": "qwen_tts_voicedesign",
            "payload": {"input": args.text_file.read_text(encoding="utf-8").strip()
                        if args.text_file and args.text_file.exists() else STYLE_INSTRUCTION,
                        "task_type": "VoiceDesign",
                        "instructions": VOICEDESIGN_INSTRUCTION, **common},
            "output": args.output_dir / "qwen_tts_ru_voicedesign.wav",
        })
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-variant", choices=tuple(MODEL_VARIANTS), default=DEFAULT_VARIANT,
                        help="Which Qwen3-TTS checkpoint to serve. Each variant only supports "
                             "some of basic/clone/style/voicedesign; unsupported jobs for the "
                             "loaded variant are reported SKIPPED, never attempted.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument("--ref-audio", type=Path, default=None)
    parser.add_argument("--ref-text", type=Path, default=None)
    parser.add_argument("--voice", default=DEFAULT_VOICE,
                        help="Predefined CustomVoice timbre name.")
    parser.add_argument("--language", default="Russian")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--server-timeout", type=float, default=1800.0)
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    parser.add_argument("--min-capability", default=None, metavar="MAJOR.MINOR",
                        help="Skip instead of attempting when the GPU compute capability is "
                             "below this value. Omitted by default: the runner attempts the "
                             "run and records the real outcome.")
    parser.add_argument("--server-arg", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--deploy-config", type=Path, default=None,
                        help="vLLM-Omni deploy YAML. Below compute 8.0 the runner defaults to "
                             "configs/qwen3_tts_turing.yaml if it exists.")
    parser.add_argument("--server-env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--request-timeout", type=float, default=900.0)
    args = parser.parse_args()

    variant = MODEL_VARIANTS[args.model_variant]
    model_id = variant["model_id"]

    report = {
        "test": "tts_qwen_cuda",
        "model": model_id,
        "model_variant": args.model_variant,
        "backend": "vllm",
        "python": platform.python_version(),
    }
    min_capability = None
    if args.min_capability:
        major, _, minor = args.min_capability.partition(".")
        min_capability = (int(major), int(minor or 0))
    runnable, reason, facts = check_requirements(min_capability)
    report.update({k: v for k, v in facts.items() if k != "cuda_capability_tuple"})

    args.metrics.parent.mkdir(parents=True, exist_ok=True)

    def write_report() -> None:
        args.metrics.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))

    if not runnable:
        report.update({"status": "SKIPPED", "reason": reason, "results": [], **peak_memory()})
        write_report()
        return

    base_url = f"http://127.0.0.1:{args.port}"
    server = None
    sampler = DeviceMemorySampler()
    # One log file per model variant: sharing a single filename across the
    # notebook's per-variant loop clobbered the previous variant's log before
    # it could be inspected (observed on a real T4 run, #52).
    log_path = args.output_dir / f"qwen_vllm_server_{args.model_variant}.log"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download

        download_started = time.perf_counter()
        model_dir = snapshot_download(model_id)
        report["model_dir"] = model_dir
        report["weights_download_seconds"] = time.perf_counter() - download_started

        server_env = os.environ.copy()
        for item in args.server_env:
            key, _, value = item.partition("=")
            server_env[key.strip()] = value.strip()
        applied = {item.partition("=")[0].strip(): item.partition("=")[2].strip()
                   for item in args.server_env}
        report["server_env"] = applied
        if applied:
            print(f"Переменные окружения сервера: {applied}")

        capability = facts["cuda_capability_tuple"]
        command = server_command(model_dir, args, capability)
        for extra in args.server_arg:
            key, _, value = extra.partition("=")
            command += [f"--{key.strip().lstrip('-').replace('_', '-')}", value.strip()]
        report["server_command"] = command
        if "--deploy-config" in command:
            report["deploy_config"] = command[command.index("--deploy-config") + 1]

        sampler.__enter__()
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log_file:
            server = subprocess.Popen(
                command, stdout=log_file, stderr=subprocess.STDOUT,
                env=server_env, start_new_session=True,
            )
            wait_for_server(base_url, server, args.server_timeout, name="vllm")
        report["server_startup_seconds"] = time.perf_counter() - started

        results = []
        for job in build_jobs(args, variant, model_dir):
            if "skipped" in job:
                results.append({"name": job["name"], "status": "SKIPPED", "reason": job["skipped"]})
                continue
            try:
                results.append({"name": job["name"], **synthesize(
                    base_url, job["payload"], job["output"], args.request_timeout)})
            except Exception as exc:
                results.append({"name": job["name"], "status": "FAILED",
                                "exception": repr(exc), "traceback": traceback.format_exc()})
        report["results"] = results
        report["status"] = "PASSED" if any(r["status"] == "PASSED" for r in results) else "FAILED"
    except Exception as exc:
        report.update({"status": "FAILED", "exception": repr(exc), "traceback": traceback.format_exc(),
                       "server_log": str(log_path), "results": report.get("results", [])})
    finally:
        terminate_server(server)
        sampler.__exit__(None, None, None)
        report.update(peak_memory())
        report.update(sampler.report())
        report.update(attention_backend_diagnostics(log_path))
        write_report()

    if report.get("status") == "FAILED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
