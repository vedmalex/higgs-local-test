#!/usr/bin/env python3
"""Load Boson's STT helpers from one pinned Hugging Face revision."""
import argparse
import importlib.util
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL_ID = "bosonai/higgs-audio-v3-stt"
REVISION = "2ffd1aa39f5a1266931e405cba12e404a9f994b2"
FILES = ("transcribe.py", "higgs_audio_collator.py")


def prepare() -> Path:
    paths = [Path(hf_hub_download(MODEL_ID, name, revision=REVISION)) for name in FILES]
    if len({path.parent for path in paths}) != 1:
        raise RuntimeError("Hugging Face helper files did not resolve to one snapshot")
    return paths[0].parent


def load_transcribe():
    directory = prepare()
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("higgs_pinned_transcribe", directory / "transcribe.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load pinned transcribe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.transcribe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.parse_args()
    print(f"Prepared {MODEL_ID}@{REVISION} helpers in {prepare()}")
