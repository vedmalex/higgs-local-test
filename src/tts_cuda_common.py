"""Shared CUDA TTS-runner helpers, used by both `src/tts_cuda.py` (Higgs) and
`src/tts_qwen_cuda.py` (Qwen3-TTS).

Both runners serve their model through a first-party stack over HTTP and must
apply the *same* anti-false-positive waveform validation before calling a job
PASSED: an HTTP 200 with a well-formed WAV of plausible duration is not
evidence of synthesis, as issue #48 demonstrated for Higgs on a T4 (a
constant `-32768` signal). Splitting a second, independently-written check
for Qwen would risk it being weaker or inconsistent; this module is the one
place that decides what counts as a real waveform.
"""
import array
import json
import platform
import resource
import subprocess
import threading
import time
import wave
from pathlib import Path

# Below this compute capability, bfloat16 has no hardware support and vLLM
# refuses to load a bf16 checkpoint with it. Shared across every backend this
# project drives through vLLM-Omni.
BF16_CAPABILITY = (8, 0)


def peak_memory() -> dict:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"peak_host_rss_bytes": rss if platform.system() == "Darwin" else rss * 1024}


class DeviceMemorySampler:
    """Samples device-wide VRAM use via nvidia-smi.

    The weights live in the server process, not in this one, so
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


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def audio_statistics(path: Path) -> dict:
    """Measure whether a WAV actually contains a varying signal.

    A server can answer 200 with a correctly sized, correctly headed WAV whose
    every sample is identical. That happened on a T4 running Higgs: the payload
    was the bytes `00 80` repeated, i.e. a constant -32768 (#48). Duration and
    byte count look healthy there, so only the samples distinguish speech from
    a dead signal.
    """
    with wave.open(str(path), "rb") as handle:
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        return {"checked": False, "reason": f"unsupported sample width {width}"}
    samples = array.array("h")
    samples.frombytes(frames[: len(frames) - len(frames) % 2])
    if not samples:
        return {"checked": True, "empty": True}
    peak = max(max(samples), -min(samples))
    rms = (sum(int(value) * int(value) for value in samples) / len(samples)) ** 0.5
    full_scale = sum(1 for value in samples if abs(value) > 32000) / len(samples)
    return {
        "checked": True, "empty": False, "samples": len(samples),
        "peak": peak, "rms": round(rms, 1),
        "full_scale_fraction": round(full_scale, 6),
        "distinct_values_in_first_4096": len(set(samples[:4096])),
    }


def audio_defect(statistics: dict) -> str | None:
    """Return why the audio is unusable, or None when it looks like a signal."""
    if not statistics.get("checked"):
        return None
    if statistics.get("empty"):
        return "the WAV contains no samples"
    if statistics["distinct_values_in_first_4096"] <= 1:
        return (f"the signal is constant (one value repeated, "
                f"peak={statistics['peak']})")
    if statistics["full_scale_fraction"] > 0.9:
        return (f"{statistics['full_scale_fraction']:.2%} of samples sit at full scale — "
                "saturated output, not speech")
    if statistics["peak"] < 64:
        return f"the signal is silent (peak={statistics['peak']})"
    return None


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
    raise TimeoutError(f"{name} did not become ready within {timeout:.0f}s")


def synthesize(base_url: str, payload: dict, destination: Path, timeout: float) -> dict:
    """POST to `/v1/audio/speech`, save the WAV, and validate it as a real signal.

    Never lets a technically-valid WAV count as PASSED without the waveform
    checks above: `HTTP 200`, a parseable WAV, or non-zero duration alone is
    not PASS for any backend driven through this module.
    """
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
    statistics = audio_statistics(destination)
    defect = audio_defect(statistics)
    result = {
        "status": "FAILED" if defect else "PASSED",
        "processing_seconds": processing,
        "audio_duration_seconds": duration,
        "rtf": processing / duration if duration else None,
        "output": str(destination),
        "output_bytes": len(audio_bytes),
        "audio_statistics": statistics,
    }
    if defect:
        # An RTF for a constant signal is a number about nothing. Keep the timing
        # visible for diagnosis, but never let the job count as a pass.
        result["reason"] = f"the server returned audio that is not speech: {defect}"
    return result


def terminate_server(server: subprocess.Popen | None) -> None:
    """Kill the server's whole process group so VRAM is freed before the next stage."""
    import os
    import signal

    if server is None or server.poll() is not None:
        return
    os.killpg(os.getpgid(server.pid), signal.SIGTERM)
    try:
        server.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(server.pid), signal.SIGKILL)
        server.wait(timeout=60)
