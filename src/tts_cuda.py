#!/usr/bin/env python3
"""CUDA TTS runner for `bosonai/higgs-tts-3-4b`.

The checkpoint ships no remote code and its `higgs_multimodal_qwen3` architecture is
not implemented in `transformers`, so there is no plain `from_pretrained` +
`generate` path. The model card documents exactly one first-party CUDA path:
SGLang-Omni (`sgl-omni serve`), whose `higgs_tts` model implementation owns the
multi-codebook decoding and the vocoder.

This runner drives that server. It never fabricates a result: an unmet requirement
is reported as SKIPPED with the reason, and a failure keeps its traceback.
"""
import argparse
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

MODEL_ID = "bosonai/higgs-tts-3-4b"
# Immutable revision of the weights this benchmark is pinned to.
REVISION = "7556c17e05201fccd9c8cc120bc216dcc7b5d561"

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
                "peak_device_vram_note": "device-wide nvidia-smi peak while the sgl-omni server was alive"}


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


def check_requirements(min_capability: tuple[int, int] | None) -> tuple[bool, str, dict]:
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
    if shutil.which("sgl-omni") is None:
        return False, (
            "the `sgl-omni` CLI is not installed; SGLang-Omni is the only documented "
            "first-party CUDA inference path for this checkpoint"
        ), facts

    capability = facts["cuda_capability_tuple"]
    if capability < ADVISORY_COMPUTE_CAPABILITY:
        facts["capability_advisory"] = (
            f"{facts['cuda_device']} reports compute capability {facts['cuda_capability']}. "
            "SGLang-Omni states no supported-hardware floor, but its pinned flash-attn-4 / "
            "flashinfer wheels target newer architectures, so this run may fail during "
            "install or server startup. Any failure is recorded with its log rather than "
            "hidden. A `triton` or `torch_native` attention backend may help: pass it "
            "through with --server-arg attention_backend=triton."
        )
    if min_capability is not None and capability < min_capability:
        return False, (
            f"{facts['cuda_device']} has compute capability {facts['cuda_capability']}, "
            f"below the --min-capability {'.'.join(map(str, min_capability))} requested "
            "for this run"
        ), facts
    return True, "", facts


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def wait_for_server(base_url: str, process: subprocess.Popen, timeout: float) -> None:
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"sgl-omni exited with code {process.returncode} before becoming ready")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(3)
    raise TimeoutError(f"sgl-omni did not become ready within {timeout:.0f}s")


def synthesize(base_url: str, payload: dict, destination: Path, timeout: float) -> dict:
    import urllib.request

    request = urllib.request.Request(
        f"{base_url}/v1/audio/speech",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        audio_bytes = response.read()
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


def build_jobs(args) -> list[dict]:
    jobs = []

    if args.text_file and args.text_file.exists():
        basic_text = args.text_file.read_text(encoding="utf-8").strip()
    else:
        basic_text = ""
    if basic_text:
        jobs.append({
            "name": "tts_basic",
            "payload": {"input": basic_text, "response_format": "wav",
                        "max_new_tokens": args.max_new_tokens},
            "output": args.output_dir / "tts_ru_basic.wav",
        })
    else:
        jobs.append({
            "name": "tts_basic",
            "skipped": f"no synthesis text: {args.text_file} is missing or empty",
        })

    jobs.append({
        "name": "tts_controls",
        "payload": {"input": CONTROL_TEXT, "response_format": "wav",
                    "max_new_tokens": args.max_new_tokens},
        "output": args.output_dir / "tts_ru_controls.wav",
    })

    if args.ref_audio and args.ref_text and args.ref_audio.exists() and args.ref_text.exists():
        reference_text = args.ref_text.read_text(encoding="utf-8").strip()
        clone_text = basic_text or CONTROL_TEXT
        jobs.append({
            "name": "tts_clone",
            "payload": {
                "input": clone_text,
                "response_format": "wav",
                "references": [{"audio_path": str(args.ref_audio.resolve()), "text": reference_text}],
                "max_new_tokens": args.max_new_tokens,
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
                             "(e.g. --server-arg attention_backend=triton).")
    parser.add_argument("--request-timeout", type=float, default=900.0)
    args = parser.parse_args()

    report = {
        "test": "tts_cuda",
        "model": MODEL_ID,
        "revision": REVISION,
        "backend": "sglang-omni",
        "python": platform.python_version(),
    }
    min_capability = None
    if args.min_capability:
        major, _, minor = args.min_capability.partition(".")
        min_capability = (int(major), int(minor or 0))
    runnable, reason, facts = check_requirements(min_capability)
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
    log_path = args.output_dir / "sgl_omni_server.log"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        # `sgl-omni serve` has no --revision flag, so the pin is applied by resolving
        # the immutable snapshot first and serving that directory.
        from huggingface_hub import snapshot_download

        download_started = time.perf_counter()
        model_dir = snapshot_download(MODEL_ID, revision=REVISION)
        report["model_dir"] = model_dir
        report["weights_download_seconds"] = time.perf_counter() - download_started

        command = ["sgl-omni", "serve", "--model-path", model_dir, "--port", str(args.port),
                   "--host", "127.0.0.1"]
        if args.mem_fraction_static is not None:
            command += ["--mem-fraction-static", str(args.mem_fraction_static)]
        for extra in args.server_arg:
            key, _, value = extra.partition("=")
            command += [f"--{key.strip().lstrip('-').replace('_', '-')}", value.strip()]
        if args.ref_audio is not None:
            # Local reference files are only readable by the server when their
            # directory is explicitly allowlisted.
            command += ["--allowed-local-media-path", str(args.ref_audio.resolve().parent)]
        report["server_command"] = command

        sampler.__enter__()
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log_file:
            server = subprocess.Popen(
                command, stdout=log_file, stderr=subprocess.STDOUT,
                env=os.environ.copy(), start_new_session=True,
            )
            wait_for_server(base_url, server, args.server_timeout)
        report["server_startup_seconds"] = time.perf_counter() - started

        results = []
        for job in build_jobs(args):
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
