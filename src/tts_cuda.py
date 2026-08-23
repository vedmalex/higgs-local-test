#!/usr/bin/env python3
"""CUDA TTS runner for `bosonai/higgs-tts-3-4b`.

The checkpoint ships no remote code and its `higgs_multimodal_qwen3` architecture is
not implemented in `transformers`, so there is no plain `from_pretrained` + `generate`
path. Two first-party serving stacks implement it, and this runner drives either:

* **vllm** (default) — `vllm serve --omni` from `vllm-omni`. Supports Python 3.13, runs
  its Higgs stages eager, and does not reach flashinfer's CuTe kernels, so it is the
  only one of the two with a plausible path on a pre-Ampere device.
* **sglang** — `sgl-omni serve`, the path named on the model card. On a Tesla T4 it
  loads the weights and then dies during CUDA graph capture with `KeyError: 'sm_75'`.

Both expose `POST /v1/audio/speech` returning WAV bytes; they differ in how a voice
reference is passed. This runner never fabricates a result: an unmet requirement is
reported as SKIPPED with the reason, and a failure keeps its traceback.
"""
import argparse
import base64
import json
import os
import platform
import resource
import shutil
import signal
import subprocess
import threading
import time
import traceback
import wave
from pathlib import Path

DEFAULT_BACKEND = "vllm"
MODEL_ID = "bosonai/higgs-tts-3-4b"
# Immutable revision of the weights this benchmark is pinned to.
REVISION = "7556c17e05201fccd9c8cc120bc216dcc7b5d561"

# Observed on a Tesla T4 (sm75): the server loads the weights, then dies during CUDA
# graph capture inside flashinfer's CUTLASS-DSL RMSNorm with `KeyError: 'sm_75'`
# (cutlass/base_dsl/arch.py). flashinfer documents `FLASHINFER_USE_CUDA_NORM=1` as the
# CUDA-JIT fallback for exactly that path, and SGLang-Omni honours a pre-set value —
# it only auto-applies it for sm100+. So the runner sets it for pre-Ampere devices.
NORM_FALLBACK_BELOW_CAPABILITY = (8, 0)

# vLLM-Omni's speech endpoint rejects a longer voice reference outright:
# "Reference audio too long (60.0s). Maximum 30s supported — use a shorter clip."
# docs/guides/voice_cloning_guide.md recommends 7-12s anyway.
MAX_REFERENCE_SECONDS = {"vllm": 30.0, "sglang": None}

# Below this, bfloat16 has no hardware support and vLLM refuses to load with it.
BF16_CAPABILITY = (8, 0)

# Below this, vLLM's own FlashInfer gate rejects the attention backend that
# vllm-omni's default Higgs deploy profile pins, aborting startup with
# "Reason: ['compute capability not supported']". This repository ships a profile
# that selects TRITON_ATTN instead, which accepts every capability.
TURING_DEPLOY_CONFIG = (Path(__file__).resolve().parents[1]
                        / "configs/higgs_multimodal_qwen3_turing.yaml")

# SGLang-Omni publishes no supported-hardware floor. Its pinned flash-attn-4 and
# flashinfer wheels target recent datacenter architectures, and `higgs_tts/sampler.py`
# calls flashinfer renorm kernels, so an older device such as Colab's T4 (compute 7.5)
# is expected to fail — but SGLang also ships `triton` and `torch_native` attention
# backends, so this is an untested expectation, not a documented requirement.
#
# Therefore the runner does NOT refuse to start below this line. It warns, attempts the
# run, and records whatever actually happens. Pass --min-capability to turn the
# expectation into a hard skip when burning GPU quota on a likely failure is not wanted.
ADVISORY_COMPUTE_CAPABILITY = (8, 9)

CONTROL_TEXT = (
    "<|emotion:contentment|><|prosody:speed_slow|>Начнём спокойно и внимательно. "
    "<|prosody:pause|> Теперь голос становится выразительнее. "
    "<|emotion:enthusiasm|><|prosody:expressive_high|>Это важная и радостная проверка! "
    "<|prosody:long_pause|><|style:whispering|>А теперь тихое завершение."
)


def peak_memory() -> dict:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"peak_host_rss_bytes": rss if platform.system() == "Darwin" else rss * 1024}


class DeviceMemorySampler:
    """Samples device-wide VRAM use via nvidia-smi.

    The weights live in the `sgl-omni` server process, not in this one, so
    `torch.cuda.max_memory_allocated()` here would report roughly zero. Only a
    device-wide sample describes what the TTS stage actually occupies.
    """

    def __init__(self, interval: float = 2.0) -> None:
        self.interval = interval
        self.peak_bytes = 0
        self.samples = 0
        self._stop = threading.Event()
        self._started = False
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self) -> None:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
        if result.returncode != 0:
            return
        used_mib = max(int(line) for line in result.stdout.split() if line.isdigit())
        self.peak_bytes = max(self.peak_bytes, used_mib * 1024 * 1024)
        self.samples += 1

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._sample()
            except Exception:
                # A sampling failure must never mask the benchmark result.
                pass

    def __enter__(self) -> "DeviceMemorySampler":
        self._thread.start()
        self._started = True
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        if self._started:
            self._thread.join(timeout=self.interval + 5)

    def report(self) -> dict:
        if not self.samples:
            return {"peak_device_vram_bytes": None,
                    "peak_device_vram_note": "nvidia-smi sampling produced no reading"}
        return {"peak_device_vram_bytes": self.peak_bytes,
                "peak_device_vram_samples": self.samples,
                "peak_device_vram_note": "device-wide nvidia-smi peak while the server was alive"}


def gpu_facts() -> dict:
    try:
        import torch
    except ImportError as exc:
        return {"gate": "torch_missing", "detail": repr(exc)}
    if not torch.cuda.is_available():
        return {"gate": "no_cuda", "torch": torch.__version__}
    props = torch.cuda.get_device_properties(0)
    return {
        "torch": torch.__version__,
        "cuda_device": props.name,
        "cuda_capability": f"{props.major}.{props.minor}",
        "cuda_capability_tuple": (props.major, props.minor),
        "cuda_total_memory_bytes": props.total_memory,
    }


def check_requirements(backend: str, min_capability: tuple[int, int] | None) -> tuple[bool, str, dict]:
    """Return (runnable, reason, facts).

    Only genuinely missing prerequisites block the run. A low compute capability
    produces an advisory, because the expectation that it fails is unverified and
    skipping would prevent ever recording what actually happens.
    """
    facts = gpu_facts()
    if facts.get("gate") == "torch_missing":
        return False, (
            "torch is missing from this environment, which means the SGLang-Omni "
            "install never completed here — `sglang-omni` pins its own torch. "
            "Check the pip output from the environment setup step"
        ), facts
    if facts.get("gate") == "no_cuda":
        return False, "no CUDA device is visible to torch", facts
    executable = "vllm" if backend == "vllm" else "sgl-omni"
    if shutil.which(executable) is None:
        return False, (
            f"the `{executable}` CLI is not installed, so the {backend} backend cannot run"
        ), facts

    capability = facts["cuda_capability_tuple"]
    if backend == "sglang" and capability < ADVISORY_COMPUTE_CAPABILITY:
        facts["capability_advisory"] = (
            f"{facts['cuda_device']} reports compute capability {facts['cuda_capability']}. "
            "SGLang-Omni states no supported-hardware floor, but its pinned flash-attn-4 / "
            "flashinfer wheels target newer architectures, so this run may fail during "
            "install or server startup. Any failure is recorded with its log rather than "
            "hidden. The runner already applies FLASHINFER_USE_CUDA_NORM=1 below "
            "compute 8.0; if graph capture still fails, try "
            "--server-arg talker-cuda-graph=off."
        )
    if min_capability is not None and capability < min_capability:
        return False, (
            f"{facts['cuda_device']} has compute capability {facts['cuda_capability']}, "
            f"below the --min-capability {'.'.join(map(str, min_capability))} requested "
            "for this run"
        ), facts
    return True, "", facts


def server_command(backend: str, model_dir: str, args, capability: tuple) -> list:
    """Command line for the chosen server, plus the flags a pre-Ampere GPU needs."""
    if backend == "vllm":
        command = ["vllm", "serve", model_dir, "--trust-remote-code", "--omni",
                   "--host", "127.0.0.1", "--port", str(args.port)]
        if capability < BF16_CAPABILITY:
            # The checkpoint declares bfloat16; vLLM refuses it below compute 8.0.
            command += ["--dtype", "float16"]
            if args.deploy_config is None and TURING_DEPLOY_CONFIG.exists():
                command += ["--deploy-config", str(TURING_DEPLOY_CONFIG)]
        if args.deploy_config is not None:
            command += ["--deploy-config", str(args.deploy_config)]
        if args.mem_fraction_static is not None:
            command += ["--gpu-memory-utilization", str(args.mem_fraction_static)]
        return command

    command = ["sgl-omni", "serve", "--model-path", model_dir,
               "--host", "127.0.0.1", "--port", str(args.port)]
    if args.mem_fraction_static is not None:
        command += ["--mem-fraction-static", str(args.mem_fraction_static)]
    if args.ref_audio is not None:
        # SGLang reads the reference from disk, so its directory must be allowlisted.
        command += ["--allowed-local-media-path", str(args.ref_audio.resolve().parent)]
    return command


def reference_payload(backend: str, ref_audio: Path, ref_text: str) -> dict:
    """Voice-reference fields, which differ between the two servers."""
    if backend == "vllm":
        mime = "audio/wav" if ref_audio.suffix.lower() == ".wav" else "audio/mpeg"
        encoded = base64.b64encode(ref_audio.read_bytes()).decode("ascii")
        payload = {"ref_audio": f"data:{mime};base64,{encoded}"}
        if ref_text:
            payload["ref_text"] = ref_text
        return payload
    return {"references": [{"audio_path": str(ref_audio.resolve()), "text": ref_text}]}


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def wait_for_server(base_url: str, process: subprocess.Popen, timeout: float,
                    name: str = "server") -> None:
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{name} exited with code {process.returncode} before becoming ready")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(3)
    raise TimeoutError(f"sgl-omni did not become ready within {timeout:.0f}s")


def synthesize(base_url: str, payload: dict, destination: Path, timeout: float) -> dict:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        f"{base_url}/v1/audio/speech",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            audio_bytes = response.read()
    except urllib.error.HTTPError as error:
        # The body says what is actually wrong ("Reference audio too long (60.0s)…").
        # Raising the bare status would hide it and send the operator to the log.
        detail = error.read().decode("utf-8", "replace").strip()
        raise RuntimeError(f"HTTP {error.code} from {error.url}: {detail}") from error
    processing = time.perf_counter() - started
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(audio_bytes)
    duration = wav_duration(destination)
    return {
        "status": "PASSED",
        "processing_seconds": processing,
        "audio_duration_seconds": duration,
        "rtf": processing / duration if duration else None,
        "output": str(destination),
        "output_bytes": len(audio_bytes),
    }


def build_jobs(args, model_dir: str) -> list[dict]:
    """One job per synthesis mode. Missing inputs become SKIPPED, never substituted."""
    jobs = []
    # vLLM validates the `model` field against what it was served with.
    common = {"response_format": "wav", "max_new_tokens": args.max_new_tokens}
    if args.backend == "vllm":
        common["model"] = model_dir

    if args.text_file and args.text_file.exists():
        basic_text = args.text_file.read_text(encoding="utf-8").strip()
    else:
        basic_text = ""
    if basic_text:
        jobs.append({
            "name": "tts_basic",
            "payload": {"input": basic_text, **common},
            "output": args.output_dir / "tts_ru_basic.wav",
        })
    else:
        jobs.append({
            "name": "tts_basic",
            "skipped": f"no synthesis text: {args.text_file} is missing or empty",
        })

    jobs.append({
        "name": "tts_controls",
        "payload": {"input": CONTROL_TEXT, **common},
        "output": args.output_dir / "tts_ru_controls.wav",
    })

    reference_limit = MAX_REFERENCE_SECONDS.get(args.backend)
    reference_seconds = None
    if args.ref_audio and args.ref_audio.exists():
        try:
            reference_seconds = wav_duration(args.ref_audio)
        except Exception:
            # Not a WAV, or unreadable: let the server judge it rather than guess.
            reference_seconds = None

    if (reference_limit is not None and reference_seconds is not None
            and reference_seconds > reference_limit):
        jobs.append({
            "name": "tts_clone",
            "skipped": (f"reference audio is {reference_seconds:.1f}s, above the "
                        f"{reference_limit:.0f}s the {args.backend} backend accepts. "
                        "docs/guides/voice_cloning_guide.md recommends 7-12s"),
        })
    elif args.ref_audio and args.ref_text and args.ref_audio.exists() and args.ref_text.exists():
        reference_text = args.ref_text.read_text(encoding="utf-8").strip()
        clone_text = basic_text or CONTROL_TEXT
        jobs.append({
            "name": "tts_clone",
            "payload": {
                "input": clone_text,
                **common,
                **reference_payload(args.backend, args.ref_audio, reference_text),
            },
            "output": args.output_dir / "tts_ru_clone.wav",
        })
    else:
        jobs.append({
            "name": "tts_clone",
            "skipped": "voice cloning needs both reference audio and its exact transcript",
        })
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("vllm", "sglang"), default=DEFAULT_BACKEND,
                        help="Serving stack. vllm (default) supports Python 3.13 and runs "
                             "the Higgs stages eager; sglang is the path named on the model "
                             "card but needs CUDA graph capture that fails on Turing.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument("--ref-audio", type=Path, default=None)
    parser.add_argument("--ref-text", type=Path, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--server-timeout", type=float, default=1800.0)
    parser.add_argument("--mem-fraction-static", type=float, default=None,
                        help="Passed through to sgl-omni to cap the static VRAM pool.")
    parser.add_argument("--min-capability", default=None, metavar="MAJOR.MINOR",
                        help="Skip instead of attempting when the GPU compute capability is "
                             "below this value. Omitted by default: the runner attempts the "
                             "run and records the real outcome.")
    parser.add_argument("--server-arg", action="append", default=[], metavar="KEY=VALUE",
                        help="Extra `sgl-omni serve` argument, repeatable "
                             "(e.g. --server-arg talker-cuda-graph=off).")
    parser.add_argument("--deploy-config", type=Path, default=None,
                        help="vLLM-Omni deploy YAML. Below compute 8.0 the runner "
                             "defaults to configs/higgs_multimodal_qwen3_turing.yaml, "
                             "because the auto-discovered profile pins an attention "
                             "backend that pre-Ampere devices reject.")
    parser.add_argument("--server-env", action="append", default=[], metavar="KEY=VALUE",
                        help="Extra environment variable for the server process, "
                             "repeatable. Overrides the runner's own defaults.")
    parser.add_argument("--request-timeout", type=float, default=900.0)
    args = parser.parse_args()

    report = {
        "test": "tts_cuda",
        "model": MODEL_ID,
        "revision": REVISION,
        "backend": args.backend,
        "python": platform.python_version(),
    }
    min_capability = None
    if args.min_capability:
        major, _, minor = args.min_capability.partition(".")
        min_capability = (int(major), int(minor or 0))
    runnable, reason, facts = check_requirements(args.backend, min_capability)
    report.update({k: v for k, v in facts.items() if k != "cuda_capability_tuple"})
    if facts.get("capability_advisory"):
        print(f"⚠️  {facts['capability_advisory']}")

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
    log_path = args.output_dir / f"{args.backend}_server.log"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        # `sgl-omni serve` has no --revision flag, so the pin is applied by resolving
        # the immutable snapshot first and serving that directory.
        from huggingface_hub import snapshot_download

        download_started = time.perf_counter()
        model_dir = snapshot_download(MODEL_ID, revision=REVISION)
        report["model_dir"] = model_dir
        report["weights_download_seconds"] = time.perf_counter() - download_started

        server_env = os.environ.copy()
        capability = facts["cuda_capability_tuple"]
        if args.backend == "sglang" and capability < NORM_FALLBACK_BELOW_CAPABILITY:
            server_env["FLASHINFER_USE_CUDA_NORM"] = "1"
        for item in args.server_env:
            key, _, value = item.partition("=")
            server_env[key.strip()] = value.strip()
        applied = {k: server_env[k] for k in ("FLASHINFER_USE_CUDA_NORM",)
                   if k in server_env}
        applied.update({item.partition("=")[0].strip(): item.partition("=")[2].strip()
                        for item in args.server_env})
        report["server_env"] = applied
        if applied:
            print(f"Переменные окружения сервера: {applied}")

        command = server_command(args.backend, model_dir, args, capability)
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
            wait_for_server(base_url, server, args.server_timeout, name=command[0])
        report["server_startup_seconds"] = time.perf_counter() - started

        results = []
        for job in build_jobs(args, model_dir):
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
        if server is not None and server.poll() is None:
            # The server holds every byte of TTS VRAM; terminating its process group is
            # what actually frees the device before the next stage.
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
            try:
                server.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(server.pid), signal.SIGKILL)
                server.wait(timeout=60)
        sampler.__exit__(None, None, None)
        report.update(peak_memory())
        report.update(sampler.report())
        write_report()

    if report.get("status") == "FAILED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
