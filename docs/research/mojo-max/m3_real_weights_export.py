"""M3-6 support script for issue #57: export the real stride-5 `_BosonDecoderBlock` (block 1)
weights already extracted by `m3_block_weights.py` (M3-3) into a plain FP32 `.npz` cache that
the pixi/MAX toolchain env (which has no `torch`) can load directly.

Why this exists (not scope creep, a hard environment-split constraint): `m3_block_weights.py`
reads the real checkpoint's BF16 tensors via `safetensors`' `framework="pt"` and upcasts with
`torch`. BF16 has no native NumPy dtype -- confirmed empirically against this exact checkpoint:
`safe_open(path, framework="numpy").get_tensor(key)` on a BF16 key raises
`TypeError: data type 'bfloat16' not understood`. Reading these tensors therefore requires
`torch`, which lives only in `.venv-tts` (M3-3/M3-4's env); `m3_decoder_block_prototype.py`
(M3-5/M3-6) runs in the separate pixi/MAX toolchain env, which has no `torch` at all (confirmed:
`pixi run python -c "import torch"` -> `ModuleNotFoundError`). This script performs the one
torch-dependent step -- load the real BF16 tensors via `m3_block_weights.load_raw`, upcast to
FP32, generate a fixed-seed random input matching `m3_block_reference.run_real_weight_cross_
check`'s pattern -- and writes everything as a plain FP32 `.npz` that needs only NumPy to read
back, so `m3_decoder_block_prototype.py --real-weights` can load it in the pixi env without ever
importing torch for real (it still *stubs* `torch` at import time only, per its existing
`_stub_torch_for_import()`, to satisfy `m3_block_reference.py`'s module-level `import torch`).

No weight-norm fold is performed here: per M3-3's finding, the real checkpoint's
`acoustic_decoder.*` conv tensors are already plain, folded `weight`/`bias` (no `weight_g`/
`weight_v` pair exists), so the extracted tensors are used directly, exactly as
`m3_block_reference.run_real_weight_cross_check` already does.

Run (must be `.venv-tts`, the only env with torch/safetensors/transformers wired up for this
checkpoint):

    .venv-tts/bin/python docs/research/mojo-max/m3_real_weights_export.py \
        [--seed 99] [--seq-len 20] [--out docs/research/mojo-max/.m3_real_weights_block1.npz]

The output path defaults to a dotfile cache next to this script (gitignored -- it is a
regeneratable derived artifact of the real checkpoint, not source; see `.gitignore`).
`m3_decoder_block_prototype.py --real-weights` invokes this script automatically (via
subprocess, `.venv-tts/bin/python`) if the cache is missing or `--seed`/`--seq-len` differ from
what is cached, so there is no separate manual step required to reproduce M3-6.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import m3_block_weights as m33  # noqa: E402  (torch/safetensors/transformers, .venv-tts only)

DEFAULT_OUT = HERE / ".m3_real_weights_block1.npz"


def export(seed: int, seq_len: int, out_path: Path) -> None:
    arrays: dict[str, np.ndarray] = {}

    for name in m33.CONV_KERNEL_NAMES:
        arrays[f"{name}.weight"] = m33.load_raw(f"{name}.weight").to(torch.float32).numpy()
        arrays[f"{name}.bias"] = m33.load_raw(f"{name}.bias").to(torch.float32).numpy()
    for name in m33.ALPHA_NAMES:
        arrays[f"{name}.alpha"] = m33.load_raw(f"{name}.alpha").to(torch.float32).numpy()

    # ConvTranspose1d layout is [C_in, C_out, K] -- C_in is the block's real input_dim.
    input_dim = int(arrays["conv_t1.weight"].shape[0])
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=(1, input_dim, seq_len)).astype(np.float32)
    arrays["x"] = x
    arrays["block_index"] = np.array(m33.BLOCK_INDEX)
    arrays["seed"] = np.array(seed)
    arrays["seq_len"] = np.array(seq_len)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)

    print(
        f"wrote {out_path} ({out_path.stat().st_size} bytes); block_index={m33.BLOCK_INDEX} "
        f"input_dim={input_dim} seq_len={seq_len} seed={seed}"
    )
    for k, v in arrays.items():
        if isinstance(v, np.ndarray) and v.ndim > 0:
            print(f"  {k}: shape={v.shape} dtype={v.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=99, help="input-generation seed (matches M3-4's real-weight cross-check default)")
    parser.add_argument("--seq-len", type=int, default=20, help="input sequence length (matches M3-5's synthetic stride-5 case)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    export(args.seed, args.seq_len, args.out)


if __name__ == "__main__":
    main()
