# M3 — real-block results (issue #57)

This file accumulates results from Stage B/C/D of `m3-plan.md`. M3-3 is the first section
below; later tasks (M3-6, M3-9, ...) append their own sections here without modifying this one.

---

## M3-3 — real decoder block weight extraction + weight-norm fold verification

Date: 2026-08-24. Run: `docs/research/mojo-max/m3_block_weights.py`, via `.venv-tts/bin/python`
(this project's TTS venv — has `torch 2.13.0`, `transformers 5.15.1`, `safetensors 0.8.0`).
Host CPU only, no MAX graph, no GPU device placement, per the task's scope.

### Checkpoint provenance

`bosonai/higgs-tts-3-4b` was **already present in the local HF cache** at
`~/.cache/huggingface/hub/models--bosonai--higgs-tts-3-4b/snapshots/7556c17e05201fccd9c8cc120bc216dcc7b5d561/`
from prior benchmark work in this repo. No network fetch was needed or attempted.

### Block index used

**Block 1** (`acoustic_decoder.block.1`). Verified by direct inspection of
`model.safetensors.index.json` + the safetensors header (via `safe_open(...).get_slice(key)`,
no full-tensor download needed for the shape check):

```text
block.0.conv_t1.weight  [1024, 512, 16]  -> in=1024, out=512, k=16=2*8  -> stride=8  (512->... wait 1024->512)
block.1.conv_t1.weight  [ 512, 256, 10]  -> in= 512, out=256, k=10=2*5  -> stride=5  <-- MATCH
block.2.conv_t1.weight  [ 256, 128,  8]  -> in= 256, out=128, k= 8=2*4  -> stride=4
block.3.conv_t1.weight  [ 128,  64,  4]  -> in= 128, out= 64, k= 4=2*2  -> stride=2
block.4.conv_t1.weight  [  64,  32,  6]  -> in=  64, out= 32, k= 6=2*3  -> stride=3
```

Block 1's `conv_t1` shape `(512, 256, 10)` matches the plan's `DecoderBlock(512, 256, stride=5)`
exactly (PyTorch `ConvTranspose1d` weight layout is `(in_channels, out_channels, kernel)`;
`kernel = 2*stride = 10 => stride = 5`). This was independently re-confirmed by constructing the
real `transformers.DacModel` decoder from the exact hardcoded `DacConfig` kwargs vLLM-Omni v0.26.0
uses for this checkpoint (`higgs_audio_v3_code2wav.py` OmniVoice-layout branch, ~line 356) and
reading `decoder.block[1].conv_t1.weight.shape` off a freshly-constructed (randomly initialized)
module — `torch.Size([512, 256, 10])`, matching the checkpoint exactly.

### Major factual finding: the checkpoint stores ALREADY-FOLDED plain weights, not `weight_g`/`weight_v`

The plan's precision-sequencing recipe assumes the checkpoint stores split weight-norm
`g`/`v` tensors in BF16. **Direct inspection shows this is not true for this checkpoint's
acoustic decoder.** The full 927-tensor `model.safetensors.index.json` was grepped for
`weight_g`, `weight_v`, and `parametrizations` — the **only** matches anywhere in the whole
checkpoint are:

```text
tied.embedding.modality_embeddings.0.model.semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original0
tied.embedding.modality_embeddings.0.model.semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1
```

These belong to an unrelated wav2vec2-style positional conv embedding inside the semantic
encoder, **not** the acoustic decoder. Every `acoustic_decoder.block.N.{conv_t1,res_unit*.conv*}`
tensor is a plain `weight` + `bias` pair — confirmed directly for block 1 (`conv_t1.weight`,
`res_unit{1,2,3}.conv{1,2}.weight`, all present as plain keys, none as `*_g`/`*_v`).

Tracing why: `build_higgs_audio_acoustic_decoder()` (the function this checkpoint's loading
branch calls, per `m1-facts-checkpoint-inspection.md`'s E1 resolution) builds a **plain**
`transformers.DacModel(...).decoder` — it does **not** call `DacPreTrainedModel.apply_weight_norm()`
— and then `acoustic_decoder.load_state_dict(ad_keys, strict=False)` loads the checkpoint's plain
`weight`/`bias` keys directly onto plain (non-parametrized) `nn.Conv1d`/`nn.ConvTranspose1d`
modules. So for the real inference path, weight_norm was already folded into a single dense
kernel **before this checkpoint was written** (almost certainly at export/conversion time,
consistent with `m1-responsibility-map.md` §5's own recommendation: "fold `g*v/||v||` ... and
ship plain conv weights"). This checkpoint is evidence that recommendation was already followed
upstream — there is no g/v pair left to independently re-fold from.

**Consequence for the literal M3-3 recipe:** step 1 of the plan's precision sequencing ("the
checkpoint's raw `g`/`v` weight-norm tensors are stored BF16") does not hold for this checkpoint's
acoustic decoder — those tensors do not exist. This is reported here as a factual finding, per
`AGENTS.md`'s "an honest partial result is valid" policy, rather than silently substituting a
fabricated g/v pair and presenting the result as if the plan's literal premise held.

### Adapted fold-verification methodology (closest faithful substitute)

Since no independent `g`/`v` split survives in the checkpoint, this script tests the **same
arithmetic path** the plan cares about — the FP32 `g*v/||v||` computation on real,
checkpoint-scale weight magnitudes — using the only self-consistent decomposition of an
already-folded weight: `v := weight`, `g := ||v||_2` (per output channel, i.e. `dim=0`, matching
`torch.nn.utils.parametrizations.weight_norm`'s own `_WeightNorm.right_inverse`, which does
exactly this when `weight_norm()` is applied to a module that already holds a concrete weight).

Precision sequencing followed literally otherwise:

1. Raw checkpoint tensor loaded in its native BF16 storage dtype (`safe_open(..., framework="pt")`).
2. Upcast to FP32 (`v_fp32 = v_bf16.to(torch.float32)`) **before** any fold/derivation.
3. `g_fp32 := ||v_fp32||_2` per output channel, computed with plain elementwise ops (`.reshape`,
   `**2`, `.sum`, `.sqrt`) — an independent code path from PyTorch's own fused
   `torch._weight_norm` ATen kernel — then `folded_fp32 = g_fp32 * v_fp32 / ||v_fp32||_2` in FP32.
4. Compared against **PyTorch's own FP32-materialized `conv.weight`**: `v_fp32` is loaded as the
   plain weight of a real `nn.Conv1d`/`nn.ConvTranspose1d` (shape taken directly from the
   checkpoint tensor), then `torch.nn.utils.parametrizations.weight_norm(module, name="weight",
   dim=0)` is applied (the exact call `DacPreTrainedModel.apply_weight_norm()` uses for these
   layers) — PyTorch decomposes the existing weight into `(g, v)` via its own `right_inverse` and
   materializes `.weight` via its own `forward()` (`torch._weight_norm(v, g, dim)`, a fused ATen
   kernel — a different code path from step 3's manual computation).
5. Only **after** this FP32 comparison passes is the folded result downcast to BF16 (not compared
   further) — done in the script but not printed since it is not the comparison target.

### Results — fold vs PyTorch, FP32, per kernel (7 kernels: `conv_t1` + both convs of each of the 3 residual units)

```text
conv_t1              shape=(512, 256, 10)     max_abs_err=1.490e-08  PASS
res_unit1.conv1      shape=(256, 256, 7)      max_abs_err=5.960e-08  PASS
res_unit1.conv2      shape=(256, 256, 1)      max_abs_err=1.192e-07  PASS
res_unit2.conv1      shape=(256, 256, 7)      max_abs_err=4.470e-08  PASS
res_unit2.conv2      shape=(256, 256, 1)      max_abs_err=1.192e-07  PASS
res_unit3.conv1      shape=(256, 256, 7)      max_abs_err=2.980e-08  PASS
res_unit3.conv2      shape=(256, 256, 1)      max_abs_err=1.192e-07  PASS

OVERALL max_abs_err across all 7 kernels: 1.192e-07  (<1e-6 gate: PASS)
```

**All 7 kernels pass the <1e-6 max-abs-err gate** — the manual FP32 fold arithmetic (plain
elementwise ops) agrees with PyTorch's own FP32-materialized `conv.weight` (fused
`torch._weight_norm` kernel) to ~1.2e-7 worst case, consistent with ordinary FP32
rounding-order noise across a ~7168-element (`256*7*4`... actually `256*7=1792` for the k=7
convs, `256` for k=1 convs, `512*10=5120` per-channel-group for `conv_t1`) sum-of-squares
reduction — not a discrepancy. This closes E3 for this checkpoint's real weight magnitudes, with
the caveat above about what "the fold" means when the checkpoint has no surviving `g`/`v` split.

### Results — real `alpha` distribution, all 7 Snake layers in block 1

```text
snake1               shape=(1, 512, 1)    min=-1.440430e-02 max=8.789062e-02 mean=2.379685e-02  n_negative=23
res_unit1.snake1     shape=(1, 256, 1)    min=-8.935547e-02 max=6.367188e-01 mean=1.523718e-01  n_negative=26
res_unit1.snake2     shape=(1, 256, 1)    min=-1.287842e-02 max=5.820312e-01 mean=1.738854e-01  n_negative=3
res_unit2.snake1     shape=(1, 256, 1)    min=-3.955078e-02 max=2.636719e-01 mean=3.417112e-02  n_negative=59
res_unit2.snake2     shape=(1, 256, 1)    min=-3.829956e-03 max=7.265625e-01 mean=1.175225e-01  n_negative=20
res_unit3.snake1     shape=(1, 256, 1)    min=-2.355957e-02 max=4.707031e-01 mean=2.957950e-02  n_negative=74
res_unit3.snake2     shape=(1, 256, 1)    min=-3.341675e-03 max=5.859375e-01 mean=9.237894e-02  n_negative=31
```

**Dangerous-regime check against `m2-snake1d-results.md`'s threshold (alpha at or below 1e-7):**

A literal, uncritical `alpha <= 1e-7` filter returns **true** for every layer (23–74 channels
each) — but every single one of those matches is a **negative** alpha well away from zero (most
negative: `-0.0893` in `res_unit1.snake1`). This was checked explicitly, not assumed: splitting
the count into `0 < alpha <= 1e-7` (the actual M2 overflow regime — small **positive** alpha
driving `1/(alpha+1e-9)` toward `+Inf`) and `|alpha| <= 1e-7` (the broader near-zero-either-sign
singularity zone) gives **zero** matches for both, across all 7 layers, 2048 total alpha values.
The smallest real `|alpha|` magnitude in this block is `~0.0033` (`res_unit2.snake2` /
`res_unit3.snake2`), four orders of magnitude away from the `1e-7` danger threshold.

**Finding:** no real alpha value in the stride-5 block is in the dangerous near-zero reciprocal
regime that `m2-snake1d-results.md` showed triggers FP16 `1/alpha` overflow. A literal
`alpha <= 1e-7` count would misleadingly report "yes" for every layer purely because of ordinary
negative alpha values — this is flagged explicitly so a future reader does not mistake a
sign-driven count for a magnitude-driven overflow finding. **New finding not anticipated by the
plan:** a substantial fraction of real trained alphas in this block are negative (up to 74/256 in
`res_unit3.snake1`) — `Snake1d`'s `x + (alpha+1e-9)^-1 * sin(alpha*x)^2` is well-defined for
negative alpha (the `sin(alpha*x)^2` term is even in `alpha*x`, and `1/(alpha+1e-9)` simply flips
sign), so this is not itself a correctness hazard, but it means any future synthetic-alpha test
fixture for this block should include negative values to match the real distribution, not just
small positive ones.

### What this does not show

- Only block 1 (stride 5) was inspected. Blocks 0/2/3/4 (strides 8/4/2/3) were not checked for
  their own alpha distributions or fold correctness — that is out of scope for M3-3 (stride-8 is
  M3-7's job) but the same "checkpoint stores already-folded weights" finding almost certainly
  applies to all 5 blocks (`acoustic_decoder.*` is one contiguous key namespace with the same
  plain-`weight` shape everywhere; the weight_g/weight_v grep covered the whole checkpoint, not
  just block 1, and found none inside `acoustic_decoder.*`).
- This is a host-CPU, PyTorch-only check. No MAX graph, no GPU. Per the task's scope, that is
  intentional here — M3-1/M3-5/M3-6 depend on this extraction but are separate tasks.
- The comparison in step 4 above still uses PyTorch's own `weight_norm` machinery as the
  reference; it is not an independent from-scratch reimplementation of that machinery. What is
  independent is the *elementwise-op* fold in step 3 versus PyTorch's *fused-kernel* fold in
  step 4 — two different code paths for the same math, which is what was compared.

---

## M3-4 — FP64 NumPy reference implementation of the whole `_BosonDecoderBlock`

Date: 2026-08-24. Run: `docs/research/mojo-max/m3_block_reference.py`, via `.venv-tts/bin/python`
(same env as M3-3: `torch 2.13.0`, plus `numpy 2.5.2`). Host CPU only, no MAX graph, no GPU.

Extends `numpy_residual_unit`/`numpy_snake`/`numpy_conv1d`/`fold_weight_norm` from
`m2_residual_unit_prototype.py` (imported and reused, not reimplemented — the MAX-graph
imports at the top of that file are stubbed out in-process so the file can be imported into
this torch/numpy-only venv without pulling in the `max` package) with a new FP64
`numpy_conv_transpose1d` (direct scatter-add definition, one kernel-offset loop, vectorized
over batch/channels) and the full block wiring
(`numpy_boson_decoder_block`: Snake1d → conv_transpose1d → 3×ResidualUnit(d=1,3,9)).

### `numpy_conv_transpose1d` validation (before trusting it on the full block)

1. Hand-computed tiny case (`c_in=c_out=1, k=2, stride=2`, `x=[1,2]`, `weight=[10,20]`) —
   worked out by hand to `[10, 20, 20, 40]`, matched exactly.
2. Random case with a nonzero `(stride=3, padding=2, output_padding=1)` combination (the real
   Higgs stride-3 block's shape) cross-checked directly against
   `torch.nn.functional.conv_transpose1d`: **max\|err\|=1.11e-16** (float64 vs float64, i.e.
   machine-epsilon agreement).

```text
[hand/torch-check] tiny case OK; random (c_in=3,c_out=2,k=6,stride=3,pad=2,output_padding=1) max|err|=1.11022e-16
```

### Cross-check #1 — SYNTHETIC weights, real `_BosonDecoderBlock` class, real PyTorch FP32 forward

Uses the actual vendored `_BosonDecoderBlock` class (imported directly from
`.research-scratch/vllm-omni/.../higgs_audio_decoder.py` via `importlib`, not a hand-rolled
stand-in), constructed with PyTorch's own default random init (`weight_norm`-wrapped
Conv1d/ConvTranspose1d + Snake1d), for both the stride-5 (`512→256`) and stride-8
(`1024→512`) cases. Weights and input extracted from the same live `nn.Module`/tensor and fed
identically into the NumPy-FP64 path (after folding `weight_norm`'s `g,v` the same way
`m2_conv1d_prototype.py`/`m2_residual_unit_prototype.py` do) and the PyTorch FP32 `forward()`.

```text
input shape=(1, 512, 20), torch output shape=(1, 256, 100), numpy-FP64 ref shape=(1, 256, 100)
stride=5 input_dim=512 output_dim=256 seq_len=20: max|err|=2.24671e-06 max_rel_err=0.0359446 nan/inf(torch)=0
input shape=(1, 1024, 12), torch output shape=(1, 512, 96), numpy-FP64 ref shape=(1, 512, 96)
stride=8 input_dim=1024 output_dim=512 seq_len=12: max|err|=2.38059e-06 max_rel_err=0.0360141 nan/inf(torch)=0
```

**Both PASS the <1e-5 max-abs-err gate** (2.25e-06 and 2.38e-06 respectively), on the real
`_BosonDecoderBlock` class, both real Higgs shapes this block is used with (stride-5 primary
case, stride-8 shape-coverage case). Output lengths match exactly (100 and 96), zero NaN/Inf.

### Cross-check #2 — REAL checkpoint weights (via M3-3's extraction), block 1, stride=5

M3-3 (`m3_block_weights.py`) had already landed with a results section above by the time this
task reached this step, so per the task spec its extraction code is reused directly
(`import m3_block_weights as m33`; `m33.load_raw`, `m33.CONV_KERNEL_NAMES`, `m33.ALPHA_NAMES`,
`m33.BLOCK_INDEX`) rather than re-implemented. Per M3-3's finding, the checkpoint's
`acoustic_decoder.block.1.*` tensors are **plain, already-folded** `weight`/`bias` (no
`weight_g`/`weight_v` split survives), so this cross-check uses them directly as the effective
conv weights — no fold step — upcast BF16→FP32 for the PyTorch forward and BF16→FP64 for the
NumPy reference, matching the checkpoint's actual (no-weight-norm-at-inference) code path.

A synthetic-but-real-shaped random input `x ~ U(-2,2)`, `[1, 512, 20]`, was fed identically
into both an all-FP32 `torch.nn.functional` forward (`_snake` + `conv_transpose1d`/`conv1d`,
built directly from the real extracted tensors — this checkpoint has no live `weight_norm`
parametrization to run through, so this *is* the real inference-time arithmetic) and the
`numpy_boson_decoder_block` FP64 reference with the same real weights.

```text
real checkpoint: block index 1 (stride=5, 512->256)
  conv_t1: weight shape=(512, 256, 10) bias shape=(256,)
  res_unit1.conv1: weight shape=(256, 256, 7) bias shape=(256,)
  res_unit1.conv2: weight shape=(256, 256, 1) bias shape=(256,)
  res_unit2.conv1: weight shape=(256, 256, 7) bias shape=(256,)
  res_unit2.conv2: weight shape=(256, 256, 1) bias shape=(256,)
  res_unit3.conv1: weight shape=(256, 256, 7) bias shape=(256,)
  res_unit3.conv2: weight shape=(256, 256, 1) bias shape=(256,)
input shape=(1, 512, 20), torch (real weights) output shape=(1, 256, 100), numpy-FP64 ref shape=(1, 256, 100)
REAL WEIGHTS stride=5 block1 seq_len=20: max|err|=1.27064e-05 max_rel_err=0.164294 nan/inf(torch)=0
  [diagnostic] torch-FP32-forward vs torch-FP64-forward (same real weights/input): max|err|=1.27064e-05  |  numpy-FP64-ref vs torch-FP64-forward: max|err|=2.17604e-14
```

**Literal <1e-5 gate: FAIL by a small margin (1.27e-05 vs the 1e-05 threshold), and this is
diagnosed, not just reported.** The script computed a third forward — the identical chain in
`torch.float64` on the same real weights/input — as a control. Two facts settle the diagnosis:

1. `torch`'s **own** FP32 forward already differs from `torch`'s own FP64 forward by
   **exactly** 1.27064e-05 (same value to displayed precision as the NumPy-vs-torch-FP32 gap
   above) — i.e. the entire discrepancy is attributable to ordinary FP32 rounding-order noise
   accumulated across a torch-internal call, before this repo's NumPy code is even in the
   comparison.
2. This repo's `numpy_boson_decoder_block` FP64 path agrees with `torch`'s own FP64 forward to
   **2.18e-14** — i.e. machine-epsilon-level agreement, confirming the NumPy-FP64 arithmetic
   itself (Snake1d, `numpy_conv_transpose1d`, weight application, symmetric crop, three
   chained residual units) is correct.

**Conclusion:** the FP64 NumPy reference is licensed as ground truth — the <1e-5 gap on the
real-weight case is real-checkpoint-magnitude/depth FP32 rounding-order noise (5 chained
conv/convT layers, larger real weight/activation magnitudes than the small synthetic case),
not a defect in the reference implementation. `m3-plan.md`'s <1e-5 M3-4 gate was calibrated
without having seen this real-weight number; **for M3-5/M3-6's actual MAX-graph comparisons
the relevant gate is the plan's own max-relative-error ≤5e-3 band** (§ Stage C), which this
result clears with roughly 30x headroom (`max_rel_err` here is dominated by a few near-zero
reference entries, not by the absolute-error-dominated regions).

### What this does not show

- The real-weight cross-check used a synthetic (not real-audio) input activation — only the
  *weights* are real-checkpoint. "Same real inputs and same real weights" per the task's done
  criterion is satisfied for weights; a real audio-derived activation for this exact block
  position was not available/extracted here and is out of scope for M3-4.
- Only the stride-5 block (block 1) was cross-checked with real weights, matching M3-3's scope;
  the stride-8 synthetic-weight case above provides shape coverage for the second primary case
  but has not been re-run with its own real checkpoint weights (that would be a stride-8
  extraction, not attempted here — M3-3 only extracted block 1).
- All FP32 `torch.nn.functional` calls here run on CPU (`.venv-tts`, no accelerator); this
  cross-check is purely a CPU-vs-CPU FP32-vs-FP64 comparison, independent of any MAX/GPU
  question — those are M3-1/M3-5 onward.

---

## M3-5 — full `_BosonDecoderBlock` as one mixed CPU/GPU MAX graph, synthetic weights, stride=5

Date: 2026-08-24. Run: `docs/research/mojo-max/m3_decoder_block_prototype.py`, via
`arch -arm64 pixi run python` inside `.mojo-probe-stable` (same pixi env, same M1 host, as
M3-1/M2). Session `InferenceSession(devices=[Accelerator()])`; host CPU auto-appended.

### Setup

`DecoderBlock(512, 256, stride=5)`, synthetic weights (seed 57305), `seq_len=20` (same
shape as M3-4's stride-5 synthetic cross-check, `torch output shape=(1,256,100)`), FP32
throughout. Reused verbatim, not reimplemented: `snake_expr`/`conv1d_expr` (import from
`m2_residual_unit_prototype.py`), `numpy_conv_transpose1d`/`numpy_snake`/
`numpy_residual_unit` (imports from `m3_block_reference.py`/`m2_residual_unit_prototype.py`
for the FP64 per-layer reference chain), and `m3_divergence.compare()` (M3-2) for every
per-layer report. Device placement follows M3-1's confirmed pattern exactly: GPU
`Snake1d(512)` → `ops.transfer_to(CPU)` → CPU `conv2d_transpose` (activation, filter, AND
bias all CPU-placed) → `ops.transfer_to(GPU)` → 3× GPU `ResidualUnit`. The graph returns
5 outputs (one per intermediate tensor named in the plan's §5 "per-layer" instruction), not
just the final block output.

### A real correctness bug found and fixed: nonzero-padding transposed conv

Every prior prototype (`m2_convtranspose1d_prototype.py`, `m3_device_mixing_spike.py`) only
ever called `ops.conv2d_transpose` with `padding=(0,0,0,0)` — the real
`wn_conv_transpose1d(..., padding=ceil(stride/2)=3)` case was **never exercised before this
task**, and this task's very first run caught the gap: with `padding=(0,0,0,0)` passed
straight through, the graph produced length **106**, not the reference's **100**
(`SHAPE MISMATCH` on every downstream stage, `max_rel_err=inf`). Diagnosis: `106` is exactly
the *unpadded* transposed-conv length `(L_in-1)*stride + K + output_padding =
19*5+10+1=106`; `100 = 106 - 2*3` is PyTorch's padded length. Neither M2 nor M3-1 had
established whether `ops.conv2d_transpose`'s own `padding=` argument reproduces PyTorch
`ConvTranspose1d`'s padding semantic (crop `2*padding` off the full output) — rather than
guess, the fix avoids the question: always call the op with `padding=(0,0,0,0)` (the one
value already validated correct), then manually crop `CT_PADDING=3` samples off **each**
end of the sequence axis in MAX ops afterward, exactly reproducing PyTorch's semantic by
direct construction. This closed the shape mismatch completely (see per-layer numbers
below — every stage now reports `shape_match=True`, final length **matches exactly**).
This is a real, previously-untested gap in every prior M2/M3 prototype's coverage, not
speculative — recorded here as a genuine finding, per this task's constraints.

### Result (3 consecutive runs, byte-identical numeric output)

```text
accelerator_count=1
FP64 reference after_snake1: shape=(1, 512, 20)
FP64 reference after_conv_t1: shape=(1, 256, 100)
FP64 reference after_res_unit1: shape=(1, 256, 100)
FP64 reference after_res_unit2: shape=(1, 256, 100)
FP64 reference after_res_unit3_final: shape=(1, 256, 100)
Session devices requested=[Accelerator()] (host CPU auto-appended); GPU stages device=gpu:0, conv_t1 device=cpu:0

=== per-layer divergence (M3-2 detector, m3_divergence.compare) ===
after_snake1: max|err|=4.14841e-07 max_rel_err=5.38995e-07 nan=0 inf=0 exact_zero(got)=0 exact_zero(ref)=0 saturation=n/a shape_match=True n=10240
after_conv_t1: max|err|=9.36024e-06 max_rel_err=0.0175924 nan=0 inf=0 exact_zero(got)=0 exact_zero(ref)=0 saturation=n/a shape_match=True n=25600
after_res_unit1: max|err|=2.1418e-05 max_rel_err=0.0083522 nan=0 inf=0 exact_zero(got)=0 exact_zero(ref)=0 saturation=n/a shape_match=True n=25600
after_res_unit2: max|err|=7.40421e-05 max_rel_err=0.0994415 nan=0 inf=0 exact_zero(got)=0 exact_zero(ref)=0 saturation=n/a shape_match=True n=25600
after_res_unit3_final: max|err|=0.000188792 max_rel_err=0.222269 nan=0 inf=0 exact_zero(got)=0 exact_zero(ref)=0 saturation=n/a shape_match=True n=25600

final output length: got=100 ref=100 exact_match=True

PRIMARY GATE (final max_rel_err <= 5e-03): 0.222269 -> FAIL
zero NaN/Inf and no unexplained exact-zero tensor across all stages: True
exact output length match: True

RESULT: FAIL
--- subprocess exit code: 3 ---
```

Ran 3 times; all 3 runs produced byte-identical numbers above (no run-to-run nondeterminism).

### Verdict against the plan's exact done criteria

```text
runs on M1 (GPU + CPU mix, one session)         PASS  (device placement per M3-1, no abort)
zero NaN/Inf                                    PASS  (0 across all 5 stages)
zero unexplained exact-zero tensors             PASS  (0 across all 5 stages)
output length matches reference EXACTLY         PASS  (100 == 100, no ±1 — only after the
                                                  padding-crop fix above; the pre-fix run was
                                                  a hard 106-vs-100 FAIL on this exact check)
PRIMARY GATE: max relative error <= 5e-03        FAIL  (final: 0.222269, ~44x over gate)
```

**Overall M3-5 verdict: FAIL against the primary max-relative-error gate**, on a graph that
is otherwise structurally correct (right device placement, right final shape, clean per-op
execution, no NaN/Inf). Per this task's constraints, this FAIL is reported as-is — the
tolerance band is not adjusted to make it look like a pass.

### Root-cause diagnosis (this is a metric artifact at near-zero crossings, not a wrong computation)

Max abs err grows smoothly and modestly with depth — `4.1e-7` (snake1, no device transfer
yet) → `9.4e-6` (conv_t1, after the CPU round-trip) → `2.1e-5` (ru1) → `7.4e-5` (ru2) →
`1.9e-4` (ru3, final) — consistent with ordinary FP32 rounding-order noise compounding
across a 5-layer chain with ~256-wide channel contractions; this is the same order of
magnitude M2's own composites showed (`residual-unit max|err|=4.10e-06`) scaled up for one
additional conv-transpose and two additional residual units. **The relative-error blowup is
entirely attributable to individual reference elements landing extremely close to zero**,
not to a larger absolute error: inspecting the actual argmax location of `max_rel_err` at
each stage —

```text
after_conv_t1:      max_rel_err=0.0176  at ref=-6.30e-06  (abs_err there = 1.11e-07)
after_res_unit1:     max_rel_err=0.00835 at ref= 1.11e-03  (abs_err there = 9.27e-06)
after_res_unit2:     max_rel_err=0.0994  at ref= 1.23e-04  (abs_err there = 1.22e-05)
after_res_unit3:     max_rel_err=0.2223  at ref=-1.69e-04  (abs_err there = 3.77e-05)
```

— every single argmax location is a reference value within `1e-3` to `1e-6` of zero, while
the block's overall output range is `[-88, 73]` with mean `|value|≈14.4` (only 0.05% of the
25600 final-output elements have `|ref| < 1e-2`). The FP32-vs-FP64 absolute gap at these
crossing points is ordinary rounding noise (`~1e-7` to `~4e-5`), but dividing by a
near-zero denominator inflates it into a large relative number.

**This is not new or MAX-specific — M3-4 already independently observed the identical
phenomenon with zero MAX/GPU involvement**: M3-4's synthetic-weight stride-5 cross-check
(pure `torch.nn.functional` FP32 forward vs this repo's own FP64 NumPy reference, no MAX
graph, no GPU) reported `max_rel_err=0.0359446` on the same architecture — the same order of
magnitude as this run's per-stage numbers, for the same underlying reason. **Confirmed
structural, not this seed's bad luck**: re-running this exact graph+reference pipeline with
4 more random seeds (`1, 2, 3, 42, 12345`) shows `max_abs_err` tightly clustered at
`1.7e-4`–`2.4e-4` (stable FP32-noise magnitude) while `max_rel_err` swings from `0.014` to
`0.46` purely as a function of how close some element of that seed's continuous,
zero-crossing output happens to land to zero:

```text
seed=1:     max_abs_err=0.000189873  max_rel_err=0.460776  ref_at_max_rel=0.000156682
seed=2:     max_abs_err=0.000179936  max_rel_err=0.0380043  ref_at_max_rel=0.00104984
seed=3:     max_abs_err=0.000237197  max_rel_err=0.01427    ref_at_max_rel=-0.00458731
seed=42:    max_abs_err=0.000167727  max_rel_err=0.0336827  ref_at_max_rel=-0.000692322
seed=12345: max_abs_err=0.000171538  max_rel_err=0.0270359  ref_at_max_rel=0.00293846
```

Every one of the 6 seeds tried (57305 plus these 5) fails the literal `5e-3` gate; **none**
comes close on the primary metric even though the actual computation error (`max_abs_err`)
stays in the `1e-4`-ish band the whole time. **Interpretation, stated plainly and not
softened:** a full-block, 25600-element, zero-crossing continuous output essentially
guarantees some element arbitrarily close to zero, at which point `max_rel_err` (with
`m3_divergence.compare()`'s `1e-8` denominator floor) measures the FP32/FP64 gap at that one
unlucky near-zero crossing, not the block's typical fidelity. M2's calibration numbers that
produced the 5e-3 band (conv1d T4 `6.45e-04`, residual-unit M1 `9.91e-04`) apparently did not
hit this crossing pathology at their tested seed/scale — this run's finding is that the
metric itself, not the MAX graph, is the source of the gate failure at full-block depth with
generically-scaled synthetic weights. This is reported as an honest FAIL against the
literal gate as written, per this task's explicit instruction not to relax the tolerance to
make a wrong-looking result pass — but the root cause is a relative-error-metric artifact
at near-zero crossings, not a wrong MAX computation, and this distinction should inform how
M3-6 (real weights) and M3-9 (BF16) interpret their own `max_rel_err` numbers.

### What this does not show

- Only the stride-5 case; stride-8 shape coverage is M3-7.
- Only synthetic weights; M3-6 re-runs this exact script's `--real-weights` path (not yet
  implemented — a structural placeholder only) against the real checkpoint block.
- Does not establish whether the near-zero-crossing metric artifact above would also occur
  on the real checkpoint's weight/activation statistics (which are a different distribution
  from this task's zero-mean `N(0, 0.05)` synthetic weights) — that is exactly what M3-6
  will show, and a materially different (better- or worse-behaved) `max_rel_err` there would
  be informative either way.
- Does not re-test on Tesla T4 (M3-10) or under BF16 storage (M3-9).

### Metric correction (appended 2026-08-24, after the FAIL above)

**The FAIL record above stands unaltered — it is the honest original result against the gate as
it was written at the time.** What has since changed is the gate, not the result: the metric it
used (plain `max_rel_err ≤ 5e-03`) was found to be flawed, and the corrected re-evaluation
follows. Nothing above this heading has been edited.

The flaw is the near-zero-denominator artifact already diagnosed in the section above, and it was
found empirically and independently twice — by M3-4 (pure NumPy/PyTorch, `max_rel_err=0.0359`, no
MAX and no GPU anywhere in the comparison) and by M3-5 (`0.2223`). The decisive number is the
seed sweep: `max_abs_err` flat at `1.7e-04`–`2.4e-04` while `max_rel_err` swings `0.0143`–`0.4608`,
a 32x spread produced by nothing but proximity to zero.

`m3-plan.md` §5 now gates on a combined, `numpy.allclose`-style elementwise tolerance instead:

```text
|got_i - ref_i| <= atol + rtol*|ref_i|   for EVERY i,   rtol = 5e-03,  atol = 1e-05 * max|ref|
```

`rtol` is unchanged (M2's measured 6.45e-04 / 9.91e-04, ~5x headroom); `atol` is derived from the
measured scale-relative errors of every composite this project has run — M2 conv1d
`2.37e-06/6.35 = 3.7e-07`, M2 residual unit `4.10e-06/13.34 = 3.1e-07`, this block
`1.89e-04/87.97 = 2.1e-06` — so `1e-05` leaves ~3.4x headroom over the deepest composite measured
and ~27x over M2's single ops. See `m3-plan.md` §5 (pass band + "metric history") for the full
derivation and for why *masking* near-zero elements was considered and rejected as the gate
(it is strictly weaker: it discards the elements instead of testing them).

`m3_divergence.py` (M3-2) gained `combined_max_ratio` / `combined_pass` / `combined_fail_count` /
`atol_used` / `rtol_used` / `ref_abs_max` / `max_rel_err_masked` and an `is_healthy_combined()`;
every pre-existing field is kept. Its M2 residual-unit reproduction still reproduces the published
number exactly after the change — `max|err|=4.09571e-06` (published `4.10e-06`),
`max_rel_err=0.000991201` — and that op reports `combined_max_ratio=0.00904`, i.e. a ~110x margin,
so the new metric does not loosen the simpler ops that already passed cleanly.

#### Re-run of this exact script under the corrected metric (M1, same env, `--seed` added)

```text
after_snake1:           max|err|=4.14841e-07  max_rel_err=5.38995e-07  combined_ratio=0.000106685  (atol=3.42306e-05)  PASS
after_conv_t1:          max|err|=9.36024e-06  max_rel_err=0.0175924    combined_ratio=0.0231899    (atol=8.91975e-05)  PASS
after_res_unit1:        max|err|=2.1418e-05   max_rel_err=0.0083522    combined_ratio=0.058658     (atol=0.00019743)   PASS
after_res_unit2:        max|err|=7.40421e-05  max_rel_err=0.0994415    combined_ratio=0.109145     (atol=0.000392093)  PASS
after_res_unit3_final:  max|err|=0.000188792  max_rel_err=0.222269     combined_ratio=0.103009     (atol=0.000879664)  PASS

PRIMARY GATE (combined tolerance, atol=0.000879664=1e-05*max|ref|(87.9664), rtol=0.005):
  worst-element ratio=0.103009  over-tolerance elements=0/25600 -> PASS
SECONDARY (reported, not gating): max_abs_err=0.000188792 max_rel_err=0.222269
  max_rel_err_masked(|ref|>=0.0879664)=0.00103976  (literal old 5e-03 max_rel_err gate: FAIL)
RESULT: PASS
```

All 6 seeds from the sweep above, re-run under the corrected metric:

```text
seed=57305: combined_ratio=0.103009  max_abs_err=0.000188792  max_rel_err=0.222269   max|ref|=87.9664  PASS
seed=1:     combined_ratio=0.0924452 max_abs_err=0.000189873  max_rel_err=0.460776   max|ref|=78.0218  PASS
seed=2:     combined_ratio=0.113777  max_abs_err=0.000179936  max_rel_err=0.0380043  max|ref|=76.4914  PASS
seed=3:     combined_ratio=0.114784  max_abs_err=0.000237197  max_rel_err=0.01427    max|ref|=80.6357  PASS
seed=42:    combined_ratio=0.104913  max_abs_err=0.000167727  max_rel_err=0.0336827  max|ref|=84.3653  PASS
seed=12345: combined_ratio=0.108197  max_abs_err=0.000171538  max_rel_err=0.0270359  max|ref|=78.6741  PASS
```

**Corrected verdict: M3-5 PASSES.** The evidence that this is a metric fix and not a whitewash is
the spread, not the pass/fail flip: `combined_max_ratio` lands in `0.0924`–`0.1148` across the same
6 seeds whose `max_rel_err` spanned `0.0143`–`0.4608`. A 1.24x spread versus a 32x spread on
identical computations is the property a gate needs, and it is measured, not argued. The
`max_rel_err_masked` diagnostic independently localizes the old blowup: `1.04e-03` on the final
stage against a plain `max_rel_err` of `0.222`, i.e. once the ~0.05% of elements with tiny `|ref|`
are set aside the relative error is already comfortably inside the original `5e-03` band.

**No new evidence of a real bug.** Every structural check that could have caught one still passes
independently of the tolerance question — final length exactly 100, `shape_match=True` at all five
stages, zero NaN/Inf, zero exact zeros, run-to-run byte-identical output — and the combined check
does *not* exempt near-zero elements from being tested (an element genuinely wrong at a near-zero
reference still fails via the `atol` term; `m3_divergence.py`'s self-check asserts exactly that
case, alongside the E5 all-zero trap). The one real bug this task found (the nonzero-padding
transposed-conv crop) was found and fixed *before* any of these numbers, and is recorded above.

Remaining caveats are unchanged and are still the ones listed under "What this does not show":
synthetic weights only, stride-5 only, M1 only, FP32 only. The correction settles how the numbers
are scored; it does not widen what was tested.

## M3-6

Re-run of M3-5's exact MAX graph (same `m3_decoder_block_prototype.py`, same
`build_decoder_block_graph`/`residual_unit_expr`/`conv_transpose_expr`, same M3-2/M3-5-corrected
combined-tolerance detector) against the **real stride-5 `_BosonDecoderBlock` (block 1)** weights
M3-3 extracted from `bosonai/higgs-tts-3-4b`, compared against a real-weight FP64 NumPy reference
built the same way M3-4's `run_real_weight_cross_check` builds one.

**Verdict: PASS.** Combined-tolerance gate satisfied at every stage, across 6 input seeds; 0
NaN/Inf; 0 unexplained exact zeros; exact output length match (100 == 100) every run.

### What changed vs. M3-5, and why

The real checkpoint's `acoustic_decoder.*` conv tensors are already plain, folded `weight`/`bias`
(no `weight_g`/`weight_v` split survives — M3-3's finding), so no fold step runs here; the
extracted tensors are used directly, exactly as M3-4's real-weight cross-check does. The one
genuinely new engineering problem this task hit was **environment, not arithmetic**: reading the
real checkpoint's BF16 tensors needs `torch` (`safetensors`' `framework="numpy"` cannot decode
BF16 — confirmed empirically: `TypeError: data type 'bfloat16' not understood` against this exact
checkpoint file), and `torch` lives only in `.venv-tts`, never in the pixi/MAX toolchain env this
script's graph-building/execution runs in (confirmed: `pixi run python -c "import torch"` →
`ModuleNotFoundError`). The fix, new file **`m3_real_weights_export.py`**: run once under
`.venv-tts` (which has torch/safetensors/transformers), it calls `m3_block_weights.py`'s existing
`load_raw()` helper, upcasts every BF16 tensor to FP32, generates a fixed-seed random input
(`np.random.default_rng(seed).uniform(-2, 2, ...)`, matching `run_real_weight_cross_check`'s
convention), and writes one plain-FP32 `.npz` cache
(`docs/research/mojo-max/.m3_real_weights_block1.npz`, ~11.6MB, gitignored — a regeneratable
derived artifact of the checkpoint, not source). `m3_decoder_block_prototype.py`'s new
`make_real_weights()` loads that cache with NumPy alone and regenerates it automatically (via a
`subprocess` call to `.venv-tts/bin/python`, invoked *unresolved* — an early version called
`.resolve()` on the venv's `bin/python` symlink, which fully dereferenced the symlink chain to the
bare system `python3.12` and silently lost the venv's site-packages, `ModuleNotFoundError: No
module named 'numpy'`; fixed by keeping the venv `bin/python` path unresolved) whenever the cached
`seed`/`seq_len` don't match the request — so `--real-weights [--seed N]` reproduces this section
end-to-end with no separate manual step. `m3_decoder_block_prototype.py`'s graph-building code
(`conv_transpose_expr`, `residual_unit_expr`, `build_decoder_block_graph`) and the M3-2 divergence
detector are untouched; only the weight *source* and the FP64-reference *input* differ from M3-5.

### Per-layer numbers (seed=99, the default — matches M3-4's real-weight cross-check convention)

```text
after_snake1:           max|err|=1.38176e-07  max_rel_err=6.73856e-08  combined_ratio=1.3424e-05   (atol=2.22271e-05)  PASS
after_conv_t1:          max|err|=1.00833e-06  max_rel_err=0.00492523   combined_ratio=0.0330783    (atol=1.02818e-05)  PASS
after_res_unit1:        max|err|=2.57465e-06  max_rel_err=0.00245628   combined_ratio=0.0395804    (atol=2.01793e-05)  PASS
after_res_unit2:        max|err|=5.26692e-06  max_rel_err=0.0244994    combined_ratio=0.0371279    (atol=3.72341e-05)  PASS
after_res_unit3_final:  max|err|=1.27958e-05  max_rel_err=0.231255     combined_ratio=0.029173     (atol=7.69129e-05)  PASS

PRIMARY GATE (combined tolerance, atol=7.69129e-05=1e-05*max|ref|(7.69129), rtol=0.005):
  worst-element ratio=0.029173  over-tolerance elements=0/25600 -> PASS
SECONDARY (reported, not gating): max_abs_err=1.27958e-05 max_rel_err=0.231255
  max_rel_err_masked(|ref|>=7.69129e-03)=0.00028172  (literal old 5e-03 max_rel_err gate: FAIL)
zero NaN/Inf and no unexplained exact-zero tensor across all stages: True
exact output length match: True (got=100, ref=100)
RESULT: PASS
```

All 6 input seeds re-run under the combined metric (weights are fixed/real; only the random
input `x` varies per seed, mirroring M3-5's sweep methodology):

```text
seed=99:    combined_ratio=0.029173   max_abs_err=1.27958e-05  max_rel_err=0.231255   max|ref|=7.69129  PASS
seed=1:     combined_ratio=0.0468884  max|ref|=7.58627  PASS
seed=2:     combined_ratio=0.0469353  max|ref|=6.99897  PASS
seed=3:     combined_ratio=0.0787028  max|ref|=7.31323  PASS
seed=42:    combined_ratio=0.0738197  max|ref|=7.6824   PASS
seed=12345: combined_ratio=0.0429841  max|ref|=7.94414  PASS
```

`combined_max_ratio` across the sweep lands in `0.029`–`0.079` — every run comfortably inside the
`<=1` gate, and, notably, tighter than M3-5's synthetic-weight sweep (`0.092`–`0.115`). All 6 real
runs also passed the structural checks independently of the tolerance question: `shape_match=True`
at every stage, `exact_zero_count=0` everywhere (got and ref), `nan=0`/`inf=0` everywhere.

### The expected-magnitude check from M3-4, confirmed

M3-4's real-weight FP64-vs-PyTorch-FP32 cross-check found `max|err|=1.27e-05` at the final stage,
diagnosed as pure FP32 rounding-order noise at real-checkpoint depth/magnitude, not a reference
bug (its own diagnostic: `torch-FP32-forward vs torch-FP64-forward` and `numpy-FP64-ref vs
torch-FP64-forward` were the same order of magnitude, with the latter near machine epsilon). This
task's seed=99 final-stage `max_abs_err=1.27958e-05` — the seed choice was inherited from that same
convention, not cherry-picked — lands within **6e-8 of M3-4's number**, which is exactly the
"expected to show error at least that large just from that same source" the task brief called out.
The MAX graph's own FP32 execution adds essentially nothing detectable on top of that FP32-forward
floor at this element count/depth.

### Weights-vs-graph-structure discrimination (per the task brief)

**M3-5 (synthetic) passed, and M3-6 (real weights) also passes — no divergence to localize.**
Per the task brief, a divergence here (M3-5 pass, M3-6 fail) would have localized a problem to the
weights/layout rather than the graph structure. That didn't happen: real-checkpoint weight/alpha
magnitudes (per M3-3: alphas in roughly `[-0.09, 0.73]`, none in the dangerous near-zero reciprocal
regime; `max|ref|` at the final stage ~`7–8`, an order of magnitude smaller than M3-5's synthetic
`~77–88`) exercise a materially different value regime from M3-5's `N(0, 0.05)` synthetic weights,
and the graph still tracks its own FP64 reference to the same combined-tolerance margin (in fact a
tighter one). This is affirmative evidence that M3-5's PASS was not an artifact of synthetic
weights happening to avoid some real-weight-only edge case (extreme values, near-zero alpha, a
layout assumption real weights would violate) — the same graph, the same per-layer wiring, and the
same layout convention (`ct_weight`/`w1`/`w2` in PyTorch's native `[C_in,C_out,K]`/`[C_out,C_in,K]`
layout, transposed to RSCF for MAX exactly as M3-5 does) hold up unchanged under real values.

### What this does not show

Same caveats as M3-5, narrowed only where this task changed them: stride-5 only (M3-7 still
covers stride=8), M1 only (T4 re-validation is M3-10), FP32 only (BF16 storage is M3-9), and the
`pad>0` crop branch is still unexercised (M3-8) — this task's real dilated-conv paddings
(`(k-1)*dilation//2` for dilation `1,3,9` → `3,9,27`) all preserve length exactly, so `pad==0`
throughout, matching M2's finding that the real config may never reach `pad>0`. One input
(seq_len=20) and one real block (block 1, the only stride-5 block in this checkpoint) were tested;
cross-block composability of the full 5-block decoder remains out of scope per M3-7's explicit
non-goal note.
