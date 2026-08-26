#!/usr/bin/env python3
"""Refs #57: quantize Higgs TTS 3 4B's language backbone to 8-bit while keeping
the bundled audio codec (``tied.embedding.modality_embeddings.0.model.*``) at
its original precision.

Why the codec must stay unquantized: it is the part of the checkpoint that
turns discrete codes back into a waveform. It is also small -- see the
measured fraction printed below (roughly 4% of total checkpoint bytes) -- so
leaving it alone costs almost nothing in size while avoiding audio-quality
risk from quantizing a component this precision-sensitive.

Root cause of the plain ``mlx_audio.convert --quantize`` failure (see issue
#57): the TTS ``Model`` class's own weight loading only recognizes language-
backbone tensors. Codec tensors under the ``tied.embedding.modality_embeddings
.0.model.`` prefix are not part of that model's parameter tree, so they are
silently dropped by ``save_model`` during conversion. The codec is loaded
independently and later, by ``HiggsAudioTokenizer.from_higgs_tts_checkpoint``,
which reads the *same* safetensors shard(s) directly by key prefix -- so this
script's job is simply to make sure those keys are still present, verbatim,
in the converted checkpoint's safetensors file.

Approach: run the standard mlx_audio converter to produce the quantized
language backbone (this drops the codec, as above), then copy the codec
tensors from the original, full-precision checkpoint into the converted
checkpoint's safetensors file and index, unmodified.

Usage:
    python3 scripts/quantize_higgs_tts.py \
        --hf-path bosonai/higgs-tts-3-4b \
        --mlx-path models/higgs-tts-3-4b-8bit \
        --q-bits 8 --q-mode affine
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.numpy import save_file

CODEC_PREFIX = "tied.embedding.modality_embeddings.0.model."


def resolve_hf_snapshot(hf_path: str) -> Path:
    """Resolve a local HF cache snapshot dir for hf_path, offline-only."""
    if Path(hf_path).exists():
        return Path(hf_path)
    cache_name = "models--" + hf_path.replace("/", "--")
    cache_root = Path(
        os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    ) / "hub" / cache_name
    snapshots = sorted(glob.glob(str(cache_root / "snapshots" / "*")))
    if not snapshots:
        raise FileNotFoundError(
            f"No local HF cache snapshot for {hf_path!r} under {cache_root}. "
            "This project never downloads multi-GB models on this machine "
            "(see AGENTS.md 'Model downloads') -- the source checkpoint must "
            "already be cached locally."
        )
    return Path(snapshots[-1])


def collect_codec_tensors(snapshot_dir: Path) -> dict:
    """Read codec-prefixed tensors verbatim (numpy, original dtype where
    safetensors' numpy view supports it -- bf16 falls back to pytorch view)."""
    index_path = snapshot_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shard_to_keys: dict[str, list[str]] = {}
        for key, shard in index["weight_map"].items():
            if key.startswith(CODEC_PREFIX):
                shard_to_keys.setdefault(shard, []).append(key)
    else:
        shard_to_keys = {
            p.name: None for p in snapshot_dir.glob("*.safetensors")
        }

    tensors: dict = {}
    for shard, keys in shard_to_keys.items():
        shard_path = snapshot_dir / shard
        with safe_open(str(shard_path), framework="pt") as f:
            iter_keys = keys if keys is not None else [
                k for k in f.keys() if k.startswith(CODEC_PREFIX)
            ]
            for k in iter_keys:
                t = f.get_tensor(k)
                tensors[k] = t
    return tensors


def report_size_fraction(snapshot_dir: Path) -> None:
    index_path = snapshot_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    files = set(weight_map.values())
    dtype_size = {
        "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "U8": 1, "F64": 8,
    }
    total_bytes = 0
    codec_bytes = 0
    for fname in files:
        path = snapshot_dir / fname
        with safe_open(str(path), framework="np") as f:
            for k in f.keys():
                sl = f.get_slice(k)
                nbytes = dtype_size.get(sl.get_dtype(), 4)
                for s in sl.get_shape():
                    nbytes *= s
                total_bytes += nbytes
                if k.startswith(CODEC_PREFIX):
                    codec_bytes += nbytes
    print(
        f"[INFO] original checkpoint: {total_bytes / 1e9:.3f} GB total, "
        f"{codec_bytes / 1e9:.3f} GB codec ({codec_bytes / total_bytes:.1%})"
    )


def merge_codec_into_converted(mlx_path: Path, codec_tensors: dict) -> None:
    st_path = mlx_path / "model.safetensors"
    index_path = mlx_path / "model.safetensors.index.json"

    with safe_open(str(st_path), framework="pt") as f:
        existing = {k: f.get_tensor(k) for k in f.keys()}

    before_bytes = os.path.getsize(st_path)
    merged = dict(existing)
    merged.update(codec_tensors)

    # save via torch->numpy roundtrip is not desired for bf16; use safetensors'
    # torch saver directly to preserve original dtypes bit-for-bit.
    import torch
    from safetensors.torch import save_file as save_file_torch

    save_file_torch(merged, str(st_path))
    after_bytes = os.path.getsize(st_path)

    if index_path.exists():
        index = json.loads(index_path.read_text())
        for k in codec_tensors:
            index["weight_map"][k] = st_path.name
        index.setdefault("metadata", {})["total_size"] = after_bytes
        index_path.write_text(json.dumps(index, indent=2))

    print(
        f"[INFO] merged {len(codec_tensors)} codec tensors into {st_path} "
        f"({before_bytes / 1e9:.3f} GB -> {after_bytes / 1e9:.3f} GB)"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-path", default="bosonai/higgs-tts-3-4b")
    ap.add_argument("--mlx-path", required=True)
    ap.add_argument("--q-bits", type=int, default=8)
    ap.add_argument("--q-mode", default="affine")
    ap.add_argument("--q-group-size", type=int, default=None)
    ap.add_argument(
        "--skip-convert",
        action="store_true",
        help="Assume mlx_path already holds a (codec-less) converted "
        "checkpoint and only merge the codec tensors in.",
    )
    args = ap.parse_args()

    snapshot_dir = resolve_hf_snapshot(args.hf_path)
    report_size_fraction(snapshot_dir)

    mlx_path = Path(args.mlx_path)

    if not args.skip_convert:
        cmd = [
            sys.executable, "-m", "mlx_audio.convert",
            "--hf-path", str(snapshot_dir),
            "--mlx-path", str(mlx_path),
            "--quantize",
            "--q-bits", str(args.q_bits),
            "--q-mode", args.q_mode,
            "--model-domain", "tts",
        ]
        if args.q_group_size is not None:
            cmd += ["--q-group-size", str(args.q_group_size)]
        print("[INFO] running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    codec_tensors = collect_codec_tensors(snapshot_dir)
    print(f"[INFO] read {len(codec_tensors)} codec tensors from source checkpoint")
    merge_codec_into_converted(mlx_path, codec_tensors)

    print("[INFO] done. Verify with:")
    print(
        "  python3 -c \"from mlx_audio.tts.utils import load; "
        f"load('{mlx_path}', model_type='higgs_audio_v3'); print('OK')\""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
