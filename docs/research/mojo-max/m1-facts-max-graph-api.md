# Raw facts: MAX graph API / vocoder building-block survey (haiku extraction)

Source: local shallow clone of `modular/modular` at `.research-scratch/modular`,
version **MAX 26.6.0.dev2026082305 / Mojo 1.1.0.dev2026082305** (nightly, post-26.5-stable —
the version verified on hardware in M0).

## Examples surveyed
`max/examples/*` and `max/kernels/examples/*` (pmpp GPU-fundamentals chapters). No vocoder/
codec/Code2Wav example anywhere. Only audio-adjacent example: `pytorch_custom_ops/whisper.py`
(mel-spectrogram feature extraction for STT, not TTS/vocoding) and
`pipelines/architectures/whisper/model.py`.

## Conv API — the key gap

- **`max.graph.ops` has Conv2d/Conv3d, but NO Conv1d/ConvTranspose1d op.** (`conv.py`,
  `conv_transpose.py`). `conv2d_transpose`'s GPU kernel has a `# TODO` marking it unimplemented
  on GPU (CPU only, per the agent's note on line ~121).
- **`max.nn` (the higher-level module layer) does have `ConvTranspose1d`** as a dataclass-style
  module: `ConvTranspose1d(length, in_channels, out_channels, dtype, stride=1, padding=0,
  dilation=1, output_padding=0, has_bias=False, permute=False, ...)`, weight shape
  `(kernel_length, out_channels, in_channels)`. Used today in the "Inkling" architecture.
- Plain **Conv1d** (non-transposed) does not appear as a first-class `max.nn` module; the closest
  existing pattern is `max.nn.state_space.varlen_causal_conv1d` — a **depthwise, causal** conv1d
  kernel built for state-space models (Qwen3.5 GatedDeltaNet), and
  `pipelines/architectures/inkling/layers/short_convolution.py` — depthwise causal conv1d with
  residual, width 4, no bias, explicitly ported from a vLLM reference. Neither is a general
  (non-causal, non-depthwise) Conv1d — Higgs's DAC decoder needs plain grouped/full Conv1d with
  symmetric padding (`kernel_size=7`, `padding=3`) and residual dilated units (dilation 1/3/9),
  which is a different shape than what exists today.
- **GroupNorm exists**: `max.graph.ops.group_norm(input, gamma, beta, num_groups, epsilon)` and
  `max.nn.norm.GroupNorm`. Higgs's decoder doesn't use GroupNorm though (it uses the custom
  Snake activation, not GroupNorm) — this op isn't actually needed for this specific port,
  noted only because it was asked about.
- No weight-normalization (`torch.nn.utils.weight_norm`-equivalent) utility was found by this
  search — Higgs's `_wn_conv1d`/`_wn_conv_transpose1d` wrap every conv in weight_norm; whether
  MAX has an equivalent, or whether weight_norm's effect must be pre-folded into plain weights
  at export time, is unresolved by this pass.

## Dtype/precision policy

- No per-operator `compute_dtype`/`accumulate_dtype` parameter in the graph op signatures found.
  Dtype behavior is governed by a global promotion rule (`graph/dtype_promotion.py`): promotion
  picks the operand with the highest category (bool<unsigned<signed<float) and largest bit
  width, and **never widens to a dtype wider than an existing operand** (performance-first) —
  i.e. MAX does not implicitly upcast for you the way "FP32 accumulation" would; that has to be
  arranged explicitly via `ops.cast(x, dtype)` before/after an op.
- One concrete accumulate-vs-store precedent exists: `max/python/max/nn/kernels.py:3915-3916` —
  for FP8 inputs, "the MFMA pipeline accumulates in f32 and stores bf16" (output dtype forced to
  bf16 when input is float8). This is the one place in the codebase where MAX documents an
  accumulate-dtype != storage-dtype policy, and it's FP8-specific, not a general mechanism.
- `graph/buffer_utils.py` (`cast_tensor_to`, `cast_tensors_to`) and `graph/ops/cast.py`
  (`ops.cast`) are the explicit, general mechanism for inserting a dtype change anywhere in a
  graph — this is almost certainly the tool a Higgs MAX port would use to implement the
  FP32-accumulation-then-FP16-storage pattern the M0 numerical suite already validated as
  necessary (RMSNorm large-input case) — by explicit `cast` calls around the sensitive op, not a
  declarative per-op flag.

## Practical implication for M1 (facts, not recommendation)

Every distinct architectural piece Higgs's Code2Wav decoder needs (Conv1d w/ symmetric padding,
weight-normed Conv1d/ConvTranspose1d, dilated residual units, Snake activation, 8-way RVQ
codebook sum) either (a) has no first-class MAX graph op yet (Conv1d/weight_norm), (b) has a
same-shaped but differently-purposed precedent (causal depthwise conv1d, usable as a reference
for writing a custom Mojo kernel but not a drop-in), or (c) is trivially expressible in existing
ops (Embedding+Linear+permute for the VQ layer, elementwise ops for Snake). This is exactly the
"missing/numerically sensitive operators need custom Mojo kernels" case the issue's revised
framing already anticipated — not a blocker, but real scope, not zero-cost reuse of an existing
MAX vocoder building block.
