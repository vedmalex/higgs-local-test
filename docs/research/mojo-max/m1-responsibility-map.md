# M1 — vLLM-Omni Code2Wav → MAX responsibility map (issue #57)

Date: 2026-08-24. Milestone M1 of issue #57. **This is a planning document.** Nothing in it has
been executed: no MAX graph has been built, no numeric comparison has been run, no weights have
been loaded. Every claim below is either a source citation or an explicitly labelled judgement
about scope.

Prior milestone: [`m0-results.md`](m0-results.md) (hardware probe, PASSED on Apple M1 and Colab
T4). Motivating failure: issue #48 — on T4 the vLLM-Omni Higgs path produces a degenerate
`-32768` waveform. The M2+ target is a falsifiable test: **same T4, same saved Talker output
tokens, same upstream weights — can a MAX/Mojo Code2Wav path produce valid speech where the
vLLM-Omni path produces silence/saturation?**

Framing per the issue: *port, don't reinvent*. vLLM-Omni is the reference/executable
specification. The deliverable is a parity strategy, not new infrastructure.

## Sources

All citations are to real clones read during this pass, not from memory:

```text
vLLM-Omni    .research-scratch/vllm-omni @ tag v0.26.0
  wrapper    vllm_omni/model_executor/models/higgs_audio_v3/higgs_audio_v3_code2wav.py
  layers     vllm_omni/model_executor/models/higgs_audio_v2/higgs_audio_decoder.py  (471 lines)
  boundary   vllm_omni/model_executor/models/higgs_audio_v3/... stage_input_processors/higgs_audio_v3.py
  config     vllm_omni/transformers_utils/configs/higgs_audio_v3.py
MAX/Mojo     .research-scratch/modular @ nightly
             MAX 26.6.0.dev2026082305 / Mojo 1.1.0.dev2026082305
```

Raw extraction notes: [`m1-facts-vllm-omni-code2wav.md`](m1-facts-vllm-omni-code2wav.md),
[`m1-facts-boson-dac-decoder.md`](m1-facts-boson-dac-decoder.md),
[`m1-facts-max-graph-api.md`](m1-facts-max-graph-api.md).

## Executive summary

The stage is small — one embedding table set, one linear, one activation, and a stack of 1-D
convolutions. It is also, in MAX terms, **almost entirely un-served by existing first-class ops**:

```text
component                     MAX status
----------------------------  ----------------------------------------------------------
VQ codebook lookup + proj      EXISTS — Embedding + Linear + permute, direct expression
8-way RVQ residual sum         EXISTS — elementwise add / accumulate loop
fc2 projection                 EXISTS — Linear with two transposes
Snake1d activation             EXISTS (composable) — sin/mul/div/add elementwise chain,
                               plus a per-channel learnable `alpha` parameter
Conv1d (full, non-causal,      NO EQUIVALENT — no `ops.conv1d`; only depthwise-causal
symmetric padding, dilated)    precedents exist. Custom Mojo kernel or Conv2d-with-
                               degenerate-height reshape.
ConvTranspose1d                CLOSE PRECEDENT — `max.nn.ConvTranspose1d` exists, but
                               `conv2d_transpose`'s GPU kernel is marked TODO/unimplemented
weight_norm                    NO EQUIVALENT — must be folded at export time
residual temporal crop         EXISTS — slice
FP32-accumulation policy       NO per-op flag — explicit `ops.cast` only
```

So: nothing here is impossible, and nothing here is free. The two genuinely load-bearing gaps
are **plain Conv1d** and **weight_norm**, and the highest-risk unknown is the **ConvTranspose1d
GPU path**, because that is the op the entire 960× upsample is built from and it is exactly the
op whose GPU kernel is flagged unimplemented upstream.

---

## 0. Talker → Code2Wav data-boundary contract

**model semantics.** The Talker emits discrete audio codes with a *delay pattern*: each codebook
`q` is offset, padded at the front with `BOC = 1024` and terminated with `EOC`.
`_revert_delay_pattern` strips `Q-1` leading BOC pads and the trailing EOC entries per codebook;
`_filter_real_code_frames` drops non-real frames. Any residual out-of-range code (`>= 1024`) is
zeroed with `torch.where`. Code2Wav itself is **deterministic given codes** — no speaker
embedding, no style vector, no text enters at this stage in v3.

**tensor shapes.**

```text
Talker emit (sync)     flat [Q * num_frames] int64, via OmniTokensPrompt
Talker emit (stream)   OmniPayloadStruct: codes.audio flat int64
                       + meta.left_context_size, meta.right_holdback_size, meta.finished
Code2Wav input         audio_codes  [B, num_codebooks=8, T]  int64 (validated, then .long())
internal transpose     rvq_codes    [8, B, T]
```

**weight names/layout.** None — this boundary is data, not parameters.

**dtype behavior.** Integer throughout. This is the one part of the pipeline where dtype cannot
be the cause of #48.

**conditioning inputs.** Only `left_context_size` / `right_holdback_size` in
`runtime_additional_information` — streaming-chunk overlap bookkeeping, not conditioning in the
acoustic sense.

**stage boundaries.** This is the M2 fixture seam. Because the boundary is integer codes with a
documented layout, **the Talker can be frozen out of the experiment entirely**: dump
`audio_codes` once from the T4 run that reproduces #48, and both the vLLM-Omni reference and the
MAX port consume the identical int64 tensor.

**output contract.**

```text
OmniOutput.multimodal_outputs["model_outputs"] = list[Tensor]  float32, CPU, shape [B, 1, T*960]
OmniOutput.multimodal_outputs["sr"]            = list[Tensor]  int32, value 24000
```

`forward_chunk` wraps `decode_codes` and trims `left_context_size * hop_length` samples from the
front and `right_holdback_size * hop_length` from the back.

**MAX equivalent/reimplementation.** No graph work needed. This is a **fixture format**, not a
component: a `.npz`/`.safetensors` pair holding `audio_codes` (int64) and the vLLM-Omni reference
waveform (float32). Reverting the delay pattern stays in Python on the host side for M2 — porting
`_revert_delay_pattern` into MAX buys nothing and adds a failure mode.

**Note on `sample_rate`.** The config declares `sample_rate=24000`, `frame_rate=25`,
`num_codebooks=8`, `codebook_size=1026` (1024 real + BOS 1024 + EOS 1025), and there is no dtype
field anywhere in `HiggsAudioV3Config`. But the bundled DAC `tokenizer_config` says
`sampling_rate: 16000` with `hop_length: 960`. `24000 / 960 = 25` matches `frame_rate=25`;
`16000 / 960 = 16.67` does not. The 16000 looks like an encoder-side/internal value. **Not
resolved by inspection** — M2's fixture must record the actual output length and derive the rate
from it rather than trusting either constant.

---

## 1. `HiggsAudioVQLayer` — codebook lookup + projection

Source: `higgs_audio_decoder.py:36-48`.

**model semantics.** One quantizer level. Integer indices → learned embedding → linear projection
into decoder width → channels-first layout.

**tensor shapes.**

```text
indices    [B, T]            int64
embedding  [B, T, 64]        codebook_dim = 64
project_out[B, T, hidden]    hidden_size comes from the loaded codec state, not from config
permute    [B, hidden, T]    .permute(0, 2, 1)
```

**weight names/layout.** `nn.Embedding(codebook_size=1024, codebook_dim=64)` → `codebook.weight`
`[1024, 64]`; `nn.Linear(64, hidden_size)` → `project_out.weight` `[hidden, 64]`,
`project_out.bias` `[hidden]`. In the bundled state dict these live under
`quantizer.quantizers.{i}.*`.

Note the mismatch worth carrying forward: the wrapper passes `codebook_dim=8` in the DAC config
block (`higgs_audio_v3_code2wav.py:~356`) while `HiggsAudioRVQ` is constructed with
`codebook_dim=64`. The 64 is the one that governs these tensors; the 8 belongs to the HF
`DacConfig` for the *other* decoder path (see §8). Do not conflate them.

**dtype behavior.** No explicit dtype. Embedding and Linear run at whatever the loaded weights
are. Indices are int64.

**conditioning inputs.** None.

**stage boundaries.** Consumes codes for codebook `i`; emits a `[B, hidden, T]` residual
contribution.

**output contract.** `[B, hidden_size, T]`, dtype = weight dtype.

**MAX equivalent — EXISTS, direct.** `ops.gather` on the codebook (or `max.nn.Embedding`), a
`Linear`, and a transpose. There is no gap here. This is the least risky component in the whole
port and a reasonable warm-up, but it retires little uncertainty because nothing about it was ever
in doubt.

---

## 2. `HiggsAudioRVQ` — 8-way residual sum

Source: `higgs_audio_decoder.py:51-77`, accumulator at line 73.

**model semantics.** `nn.ModuleList` of 8 `HiggsAudioVQLayer`s. `decode` allocates a zero
accumulator and **sums** all eight decoded contributions. This is residual vector quantization:
sum, **not** concatenation. Getting this wrong produces a plausible-shaped tensor with wrong
content — a silent-corruption class of bug.

**tensor shapes.** `codes [8, B, T]` → per-level `[B, hidden, T]` → summed `[B, hidden, T]`.

**weight names/layout.** `quantizer.quantizers.{0..7}.codebook.weight`,
`quantizer.quantizers.{i}.project_out.{weight,bias}`.

**dtype behavior.** **The one explicit dtype in the entire layer file**:
`torch.zeros(..., dtype=torch.float32)` at line 73. The accumulator is FP32 regardless of what
the codebooks are stored in. This is upstream already doing, by hand, exactly the
accumulate-wider-than-storage pattern M0 found necessary — and it is a constraint the MAX port
must reproduce deliberately, because MAX will not do it implicitly (see §9).

**conditioning inputs.** None.

**stage boundaries.** RVQ output feeds `fc2`.

**output contract.** `[B, hidden_size, T]` float32 (accumulator dtype), immediately re-cast by
the wrapper to `self.fc2.weight.dtype`.

**MAX equivalent — EXISTS.** Eight §1 subgraphs plus adds, or a single fused gather over a
stacked `[8, 1024, 64]` codebook followed by a reduction over the level axis. The FP32
accumulation must be written explicitly: `ops.cast(level_out, DType.float32)` before each add,
and one `ops.cast` back afterwards. **Do not let the accumulation inherit an FP16 storage
dtype** — M0's RMSNorm finding (FP16 accumulation silently zeroing an entire tensor with zero
NaN/Inf detected) is the same failure shape as #48's `-32768`.

---

## 3. `fc2` projection

Source: `higgs_audio_v3_code2wav.py` — `nn.Linear(fc2_w.shape[1], fc2_w.shape[0])`, weights
copied from `fc2.weight` / `fc2.bias` in the bundled codec state.

**model semantics.** Single affine map from RVQ hidden width to the DAC decoder's input channel
count (256 by `BosonDacDecoder`'s defaults). Bridges quantizer space to vocoder space.

**tensor shapes.**

```text
in   [B, hidden, T]  --transpose(1,2)-->  [B, T, hidden]
                     --Linear-->          [B, T, 256]
                     --transpose(1,2)-->  [B, 256, T]
```

**weight names/layout.** `fc2.weight` `[out, in]`, `fc2.bias` `[out]`. Shapes are **read from the
checkpoint**, not from config — the port must do the same rather than hardcoding.

**dtype behavior.** The wrapper aligns the incoming tensor with
`quantized.to(dtype=self.fc2.weight.dtype)` — i.e. the checkpoint's storage dtype dictates the
compute dtype here. If the checkpoint is FP16, this projection runs FP16 with no accumulation
guarantee. **This is a candidate site for #48** and worth instrumenting in M2, not just porting.

**conditioning inputs.** None.

**stage boundaries.** Last op before the conv stack.

**output contract.** `[B, 256, T]`, then re-cast to the acoustic decoder's first-parameter dtype.

**MAX equivalent — EXISTS.** `max.nn.Linear` + two transposes; or skip the transposes entirely by
folding the projection into a 1×1 conv / a `matmul` over the channel axis. Prefer folding: the
transposes exist in PyTorch only because `nn.Linear` wants channels-last.

---

## 4. Snake1d activation

Source: `_snake` (`higgs_audio_decoder.py:80-85`, `@torch.jit.script`) and `_Snake1d`
(`88-99`).

**model semantics.** Per-channel *learnable* periodic activation:

```text
snake(x) = x + (alpha + 1e-9)^-1 * sin(alpha * x)^2
```

`alpha` is a parameter of shape `[1, channels, 1]`. The implementation reshapes to `[B, C, -1]`,
applies, and reshapes back. This is the DAC family's signature activation — it is what makes the
vocoder able to represent periodic waveforms, and it is **not** any standard activation. A
substitution here (Snake → SiLU, "close enough") would not produce degraded speech; it would
produce a different model.

**tensor shapes.** `[B, C, T]` in, `[B, C, T]` out. `alpha` `[1, C, 1]`, broadcast.

**weight names/layout.** One `alpha` per Snake instance. Instance count: 1 pre-block +
per-decoder-block (1 in the block head + 2 per residual unit × 3 units) + 1 final. Under
`BosonDacDecoder`'s `nn.Sequential`, these are positional names (`model.N....alpha`); under the
HF `DacModel` layout they are named (`block.N.snake1.alpha` etc.). See §8 — the name mapping
differs entirely between the two candidate decoders.

**dtype behavior.** No explicit dtype; runs at parameter dtype. **Numerically the most dangerous
elementwise op in the stage**: `1/(alpha + 1e-9)` is a reciprocal of a learned value that can be
small, and `sin(alpha*x)^2` is bounded but the reciprocal is not. In FP16, a small `alpha`
sends the reciprocal toward the 65504 ceiling. M0 established that FP32 *accumulation* does not
rescue a value whose true magnitude exceeds the FP16 **storage** range — so if `alpha` is small
anywhere in a real checkpoint, this expression must be evaluated in FP32 and stored back down,
not merely accumulated in FP32.

**conditioning inputs.** None (`alpha` is a weight, not conditioning).

**stage boundaries.** Interleaved throughout the conv stack.

**output contract.** Same shape, same dtype as input.

**MAX equivalent — EXISTS (composable), and this is the recommended first prototype.** Every
primitive needed is present: `ops.sin`, `ops.mul`, `ops.div`, `ops.add`, `ops.pow` (or a mul by
itself), broadcasting over the channel axis. No custom Mojo kernel is required for correctness.
A fused custom Mojo kernel is a *later* performance option (M4), not an M2 requirement.
Precision handling: cast the whole expression to FP32, compute, cast back — explicit, per §9.

---

## 5. `_wn_conv1d` — weight-normalized Conv1d

Source: `higgs_audio_decoder.py:102-111` — thin wrappers,
`weight_norm(nn.Conv1d(...))` / `weight_norm(nn.ConvTranspose1d(...))`, using
`torch.nn.utils.weight_norm`.

**model semantics.** Two orthogonal facts here, and they must be separated:

1. **Weight normalization** is a *reparameterization*: the stored parameters are `weight_g`
   (magnitude) and `weight_v` (direction), and the effective kernel is
   `W = g * v / ||v||`. At inference this is a pure function of the stored weights — there is no
   runtime state, no batch dependence, nothing data-dependent.
2. **Plain 1-D convolution**, non-causal, symmetric padding, with dilation. Concretely, the
   shapes used: `k=7, padding=3` (the entry and exit convs), `k=7, dilation=d,
   padding=(7-1)*d//2` with `d ∈ {1,3,9}` (residual units), and `k=1` (the pointwise conv in each
   residual unit).

**tensor shapes.** `[B, C_in, T]` → `[B, C_out, T]` (padding is chosen to preserve `T` in every
non-transposed conv in this decoder).

**weight names/layout.** Under weight_norm each conv contributes `weight_g` `[C_out, 1, 1]` and
`weight_v` `[C_out, C_in, k]`, plus `bias` `[C_out]`. **A MAX port will not see these names if
weight_norm is folded at export**, which is what should happen — see below.

**dtype behavior.** Parameter dtype; no explicit cast. Note that folding weight_norm involves an
L2 norm over `C_in * k` elements — for `C_in=1024, k=7` that is 7168 squares summed. **Fold in
FP32 on the host, at export time**, never in FP16 at runtime; this is precisely the
sum-of-squares overflow pattern M0 caught in FP16 RMSNorm.

**conditioning inputs.** None.

**stage boundaries.** Internal to the conv stack.

**output contract.** `[B, C_out, T]`.

**MAX equivalent — this is the real gap.**

- **weight_norm: NO EQUIVALENT in MAX.** No `weight_norm` utility was found in the nightly
  clone. **Recommendation: do not port it at all.** Fold `g * v / ||v||` into a single dense
  kernel during host-side weight conversion, in FP32, and ship plain conv weights to the graph.
  This is not a workaround — it is what weight_norm reduces to at inference, and it removes a
  whole class of divergence from the parity comparison. It does mean the MAX checkpoint is a
  *derived* artifact, so M2's fixture must verify the fold itself (compare folded `W` against
  PyTorch's `conv.weight` after a `weight_norm` forward hook) as a separate, earlier check than
  the conv numerics.
- **Conv1d: NO first-class MAX graph op.** `max.graph.ops` has `conv2d`/`conv3d` but no
  `conv1d`. The nearest existing things are `max.nn.state_space.varlen_causal_conv1d` (built for
  Qwen3.5 GatedDeltaNet) and `pipelines/architectures/inkling/layers/short_convolution.py` —
  both **depthwise and causal**, width-4, no bias. Higgs needs **full (non-depthwise),
  non-causal, symmetrically padded, dilated** conv. These precedents are useful as *reference
  Mojo* for memory layout and tiling idiom; they are **not drop-in**, and describing them as
  "MAX already has conv1d" would be wrong.

  Two viable routes, to be decided by measurement in M2/M3, not by preference:

  ```text
  route A  express Conv1d as ops.conv2d with a degenerate height axis
           [B,C,T] -> [B,C,1,T], kernel [C_out,C_in,1,k], padding (0,3), dilation (1,d)
           pro: zero new kernels, uses a mature, GPU-supported op
           con: depends on conv2d handling H=1 efficiently; a layout/perf question, not
                a correctness one
  route B  custom Mojo conv1d kernel, using the varlen_causal_conv1d and
           short_convolution sources as structural references
           pro: full control over padding/dilation/accumulate dtype
           con: real work, and it is the component most likely to harbour a subtle
                off-by-one in dilated symmetric padding
  ```

  **Route A first.** It is strictly cheaper to falsify, and if it is numerically correct then
  route B becomes an M4 performance question rather than an M2 correctness dependency.

---

## 6. `_wn_conv_transpose1d` — the upsampling op

Same source lines; used only inside `_BosonDecoderBlock`.

**model semantics.** Learned temporal upsampling by `stride`. Per block:
`kernel_size = 2*stride`, `padding = ceil(stride/2)`, `output_padding = stride % 2`.

**tensor shapes.** `[B, C_in, T]` → `[B, C_out, ~T*stride]`. Across the five blocks the total
factor is `8*5*4*2*3 = 960`, matching `hop_length` exactly.

**weight names/layout.** `weight_g` / `weight_v` / `bias` as in §5. Note MAX's
`ConvTranspose1d` expects weight shape `(kernel_length, out_channels, in_channels)` whereas
PyTorch's `ConvTranspose1d` stores `(in_channels, out_channels, kernel)` — **a transpose, not a
reshape**, is required at conversion. Getting this wrong yields a running graph with garbage
audio, which is exactly the failure mode #48 already looks like; the M2 fixture must therefore
distinguish "wrong layout" from "wrong precision" rather than just reporting "output is bad".

**dtype behavior.** Parameter dtype.

**conditioning inputs.** None.

**stage boundaries.** Internal.

**output contract.** Upsampled `[B, C_out, T*stride + adjustment]`. The exact output length
depends on `padding`/`output_padding` and is the thing most likely to disagree between PyTorch
and MAX by ±1 sample per block — compounding over five blocks. `adjust_conv_transpose_output_padding`
exists on the HF path (§8) specifically to fix such an off-by-one, which is direct evidence that
this is a real hazard upstream and not a hypothetical one.

**MAX equivalent — CLOSE-BUT-IMPERFECT PRECEDENT, and the single highest-risk item.**
`max.nn.ConvTranspose1d` exists as a real module (used by the "Inkling" architecture) with
`stride`, `padding`, `dilation`, `output_padding`, `has_bias`, `permute`. That is genuinely
encouraging. **But** `conv2d_transpose`'s GPU kernel carries a `# TODO` marking it unimplemented
on GPU (CPU only). If `ConvTranspose1d` lowers through that path, the entire upsample stack may
be CPU-only on the T4 — which would not block a *correctness* parity experiment (the whole point
of M2 is numerics, not throughput) but would wreck an M4 performance story and must be known
before anyone promises a fast MAX vocoder.

**Tested 2026-08-24 on M1/Metal — worse than "unimplemented," see
[`m2-convtranspose1d-results.md`](m2-convtranspose1d-results.md).** `ops.conv2d_transpose` on
CPU works cleanly for all five of Higgs's actual `(stride, output_padding)` pairs, including both
`output_padding=1` cases — the "only 0 supported" docstring text is not enforced here. On the
Apple GPU (`Accelerator()`), every case is a **fatal process abort**, not a catchable exception:
`symbol not found: cudnnCreate` — the GPU dispatch path unconditionally tries to load NVIDIA's
cuDNN library regardless of the actual accelerator backend, and Metal has no such symbol. This
is a Metal-specific finding, not necessarily a T4/CUDA one — the crash is plausibly specific to
attempting a CUDA library load on a non-CUDA GPU, so a real T4 run (which has genuine cuDNN
available) is still the open, undetermined data point, not resolved by this M1-side test. If T4
also fails, the whole ConvTranspose1d path is CPU-only for now, on both targets; if T4 works,
this is a Metal-backend bug worth reporting to Modular.

---

## 7. `_BosonResidualUnit`, `_BosonDecoderBlock`, `BosonDacDecoder`

Sources: `higgs_audio_decoder.py:114-130`, `133-154`, `157-207`.

**model semantics.**

```text
_BosonResidualUnit(dim, dilation=d)
    y = Snake1d(dim)
      -> wn_conv1d(dim, dim, k=7, dilation=d, padding=(7-1)*d//2)
      -> Snake1d(dim)
      -> wn_conv1d(dim, dim, k=1)
    pad = (x.shape[-1] - y.shape[-1]) // 2
    if y shorter:  x = x[..., pad:-pad]        # symmetric crop of the skip path
    return x + y

_BosonDecoderBlock(input_dim, output_dim, stride)
    Snake1d(input_dim)
      -> wn_conv_transpose1d(input_dim, output_dim,
                             kernel_size=2*stride, stride=stride,
                             padding=ceil(stride/2), output_padding=stride % 2)
      -> ResidualUnit(output_dim, dilation=1)
      -> ResidualUnit(output_dim, dilation=3)
      -> ResidualUnit(output_dim, dilation=9)

BosonDacDecoder(input_channel=256, channels=1024, rates=(8,5,4,2,3), d_out=1)
    wn_conv1d(256, 1024, k=7, padding=3)
    DecoderBlock(1024, 512, stride=8)
    DecoderBlock( 512, 256, stride=5)
    DecoderBlock( 256, 128, stride=4)
    DecoderBlock( 128,  64, stride=2)
    DecoderBlock(  64,  32, stride=3)
    Snake1d(32)
    wn_conv1d(32, 1, k=7, padding=3)
    # pure nn.Sequential, no skip around the whole stack
    # hop_length = 8*5*4*2*3 = 960  (matches the wrapper's hardcoded 960 exactly)
```

**tensor shapes.** `[B, 256, T]` → `[B, 1, T*960]`. Total: 1 entry conv, 5 transposed convs,
30 residual-unit convs (5 blocks × 3 units × 2 convs), 1 exit conv = **37 convolutions**, and
1 + 5 + 30 + 1 = **37 Snake activations**.

**weight names/layout.** `BosonDacDecoder` is a bare `nn.Sequential`, so its state-dict keys are
**positional**: `model.0.*`, `model.1.block.*`, … This positional layout is exactly what
`_load_from_bundled_state` sniffs for (see §8).

**dtype behavior.** No explicit dtype anywhere in the class. The whole stack runs at whatever
dtype `load_state_dict` produced. The wrapper aligns its input with
`quantized.to(dtype=first_param.dtype)` where `first_param = next(self.acoustic_decoder.parameters())`
(`higgs_audio_v3_code2wav.py:459`). **So the entire vocoder's compute dtype is decided by the
first parameter of the loaded checkpoint** — nothing in the Higgs code chooses it. If the
checkpoint is FP16, 37 convolutions and 37 reciprocal-bearing Snakes run in FP16, and per M0
that is a regime where silent zeroing/saturation is demonstrated, not speculative. This is the
strongest structural hypothesis available for #48, and M2 should be built to test it.

**conditioning inputs.** None. `build_boson_dac_decoder(device)` instantiates with
**all defaults** — the channel counts and rates above are load-bearing hardcoded values, read
from no config field at all (`higgs_audio_decoder.py:47-49`).

**stage boundaries.** Whole-vocoder. Input `[B,256,T]` from `fc2`; output raw PCM.

**output contract.** `[B, 1, T*960]`. The wrapper unsqueezes to 3-D if needed, forces
`.to(torch.float32)`, and returns on `codes.device`.

**MAX equivalent/reimplementation.** Assembly is unremarkable once §4–§6 exist: a Python-side
graph builder mirroring the loop, since MAX graphs are constructed in Python. Two specifics
worth pinning down now:

- **The residual crop is a `slice`, not a pad.** `pad = (len_x - len_y) // 2` then
  `x[..., pad:-pad]`. Note that when `len_x - len_y` is odd, integer division makes the crop
  asymmetric-by-one in PyTorch's favour, and when `pad == 0` the expression `x[..., 0:-0]`
  would be empty — PyTorch avoids this because the branch is guarded on `y` being shorter.
  **A MAX port must reproduce the guard, not just the slice.** This is a small, high-probability
  source of shape divergence.
- **37 convolutions is enough depth for error to compound.** Per-layer parity tolerance must be
  tight (relative error, against an FP32 PyTorch reference) or a per-layer "close enough" will
  integrate into audible garbage by the output. Compare layer-by-layer, not just end-to-end.

---

## 8. OPEN QUESTION (a) — which decoder does the real v3 checkpoint load?

**Status: RESOLVED 2026-08-24, by direct inspection of the real `bosonai/higgs-tts-3-4b`
checkpoint (no download of weights needed — HTTP range request on `model.safetensors`'s own
header). Full detail: [`m1-facts-checkpoint-inspection.md`](m1-facts-checkpoint-inspection.md).**

The real checkpoint's codec keys (`acoustic_decoder.block.N.conv_t1`, `res_unit{1,2,3}.conv{1,2}`,
`snake{1,2}.alpha`, none starting with `"model."`) confirm `is_boson_layout = False` — **the real
v3 checkpoint takes the HF `DacModel` branch, not `BosonDacDecoder`**, exactly as upstream's
source comment asserted. **The weight-name tables in §5–§7 above describe `BosonDacDecoder`'s
flat positional layout and are the WRONG architecture for a port** — the real layout has one
level of named `block.N.{conv_t1,res_unit1,res_unit2,res_unit3,snake1}` nesting. A port must
target the HF `DacModel`/`decoder` architecture, not `BosonDacDecoder`.

Also newly found by the same inspection: the real quantizer keys include `project_in` and
EMA-training buffers (`embed_avg`, `cluster_size`, `inited`) that `HiggsAudioVQLayer` (the class
`_load_from_bundled_state` actually builds) does not have fields for — the wrapper's loader must
be dropping/remapping these, not doing a 1:1 rename. Not fully traced; flagged for the port's
weight-loader implementation.

**Superseded text below** (kept for the record of what this pass originally didn't know; the
mechanism description is still accurate, the "not resolved" framing is not):

---

`build_boson_dac_decoder` and `build_higgs_audio_acoustic_decoder` are two *independent* decoder
implementations (`higgs_audio_decoder.py:47` and `:50-56`). The second builds a HuggingFace
`transformers.DacModel` from a `DacConfig`, takes its `.decoder` submodule, calls
`adjust_conv_transpose_output_padding(decoder)`, and replaces `decoder.tanh` with
`nn.Identity()` if present.

Reading `higgs_audio_v3_code2wav.py:346-371` resolves *how* the choice is made — it is
**runtime key-layout sniffing**, not configuration:

```python
# Build acoustic decoder — detect layout from key names.
# V3 checkpoint uses OmniVoice layout (block.N/conv1/conv2/snake1),
# not boson-ai standalone layout (model.0/model.1).
is_boson_layout = any(k.startswith("model.") for k in ad_keys)
if is_boson_layout:
    acoustic_decoder = build_boson_dac_decoder(device)
else:
    acoustic_decoder = build_higgs_audio_acoustic_decoder(tokenizer_config, device)
```

Upstream's own comment asserts the V3 checkpoint takes the **`else` branch — the HF `DacModel`
path**. If that comment is accurate, then §7's `BosonDacDecoder` is *not* the architecture a MAX
port should target, and the entire weight-name map above (positional `model.N.*`) is the wrong
one (named `block.N.conv1/conv2/snake1.*` would be right).

**Why this is still open.** The comment is an assertion in source, not an observation of a
checkpoint. Nothing in this pass loaded real weights. Two things must be true before M2 commits:

1. Dump the actual `acoustic_decoder.*` keys from the real v3 codec state and check
   `any(k.startswith("model."))` empirically. One line of Python; **do this first**.
2. If the HF path wins, then `adjust_conv_transpose_output_padding` must be read and understood,
   because it mutates `output_padding` and therefore changes output lengths — the exact hazard
   flagged in §6. It was not traced in this pass.

Also note the two decoders may not be numerically identical even given the same weights (the
`tanh → Identity` substitution is evidence they differ in at least one place). **Porting the
wrong one is the largest single wasted-effort risk in M2.**

---

## 9. Precision policy — carried forward from M0, revised after checkpoint inspection

**Important correction, 2026-08-24**: the real `bosonai/higgs-tts-3-4b` checkpoint stores its
entire codec stack (every conv, every Snake `alpha`, every RVQ projection) in **BF16**, not
FP16 — confirmed by reading the actual `model.safetensors` tensor headers (see
[`m1-facts-checkpoint-inspection.md`](m1-facts-checkpoint-inspection.md)), sampled across the
codec and the LLM backbone alike. BF16 has FP32's exponent range (8 bits) and does not suffer
FP16's ±65504 overflow ceiling — so **M0's FP16-overflow finding does not directly explain a
BF16 checkpoint's behavior on T4**, and the paragraphs below (written before this inspection)
should be read as "what M0 established about FP16 specifically," not as the confirmed mechanism
of #48. Two hypotheses now rank above pure FP16-style overflow, neither resolved yet: (1) a
genuine BF16-on-Turing correctness gap in the specific ops used (not just a performance gap —
T4 has no BF16 tensor cores), or (2) an FP16 intermediate introduced elsewhere in the pipeline by
vLLM-Omni's `--dtype float16` server flag (used on T4 because vLLM refuses BF16 checkpoints on
pre-Ampere hardware for the LLM stage), propagating into Code2Wav's *input* even though the
codec's own weights stay BF16. E2 must capture actual runtime dtypes at the Talker→Code2Wav
boundary on a real T4 run to settle this, not assume either dtype.

The general MAX precision-mechanism facts below are unaffected by this correction — only the
"which dtype is actually at risk" conclusion changes.

**MAX has no per-operator `compute_dtype` / `accumulate_dtype` parameter.** Dtype behavior is
governed by a global promotion rule (`graph/dtype_promotion.py`) that picks the highest-category,
widest operand and **never widens beyond an existing operand** — performance-first. MAX will
*not* implicitly upcast the way "FP32 accumulation" implies. The only general mechanism is
explicit: `ops.cast` (and `graph/buffer_utils.py`'s `cast_tensor_to` / `cast_tensors_to`).

The one accumulate≠store precedent in the codebase is FP8-specific
(`max/python/max/nn/kernels.py:3915-3916` — MFMA accumulates in f32, stores bf16). It is not a
general mechanism and should not be cited as one.

Consequences for this port, applying M0's findings:

```text
M0 finding                                        Code2Wav consequence
------------------------------------------------  -----------------------------------------------
FP32 accumulation reduces but does NOT repair      Snake1d's 1/(alpha+1e-9) and weight_norm's
overflow when the TRUE value exceeds FP16's        ||v|| must be FP32 in STORAGE too, not just
finite range (65504)                               accumulation. Cast the tensor, not the reduce.
FP16 RMSNorm silently zeroed its whole output      A NaN/Inf scan will pass a fully broken
with ZERO NaN/Inf detected; FP32 accumulation      Code2Wav. M2 instrumentation must compare
fully repaired it                                  against an FP32 reference and flag all-zero
                                                   and saturated tensors per layer.
BF16 matmul "PASSED" on M1 and T4, but this is     Do not design the port around BF16 on T4.
NOT proof of hardware BF16 on Turing (sm_75)       Default to FP32 for the parity run; FP16 with
                                                   explicit FP32 casts is the optimization, later.
```

**Recommended M2 policy: build the MAX Code2Wav graph in FP32 end-to-end first.** The experiment
is "can a MAX path produce valid speech from these weights", not "can it do so quickly". An FP32
graph removes precision as a confound; if FP32 MAX produces good audio from the same weights that
give vLLM-Omni `-32768`, that is already a decisive result and it *localizes the bug to
precision*. Introduce FP16 afterwards, one region at a time, to find exactly where it breaks —
which is a far better outcome than starting mixed-precision and having two unknowns.

## 10. OPEN QUESTION (b) — is MAX's BF16 pass on T4 real hardware execution?

**Status: UNRESOLVED, inherited from M0, and NOT blocking for M2 under the FP32-first policy
above.**

M0 recorded `[PASSED] GPU matmul BF16 supported` on a Tesla T4. T4 is `sm_75` (Turing) and has no
BF16 tensor cores — those arrived with Ampere (`sm_80`). The probe only verified that a
`bfloat16`-storage matmul with FP32 accumulation compiles, runs, and produces finite output. That
cannot distinguish genuine hardware BF16 from MAX transparently upcasting to FP32 (or another
supported path) underneath. Both produce an identical PASS.

What it actually establishes: **MAX will let you declare a `bfloat16` tensor on a T4 without
erroring.** That is a compatibility fact, not a performance or implementation fact.

Resolution path (M4's territory, per M0): MAX-side documentation of its BF16-on-pre-Ampere
fallback, or a profiling pass comparing BF16 vs native FP16 throughput on the same T4.

It is listed here because if M2 or M3 ever reaches for BF16 as a "safe middle ground" between
FP16's range problems and FP32's cost, **that reach is currently unjustified on T4** and must be
gated on this question.

---

## 11. M2 entry criterion

M2 is: *build a parity fixture — same weights, same tokens — and compare vLLM-Omni against MAX
numerics.* It can start when all of the following are true. These are checks, not deliverables;
none is more than a few hours.

```text
E1  DONE 2026-08-24. Decoder identity resolved (§8): real checkpoint uses the HF
    DacModel branch, block.N-nested key layout, not BosonDacDecoder's flat
    positional layout. See m1-facts-checkpoint-inspection.md.

E2  Reference fixture captured on T4.
    From the run that reproduces #48: save audio_codes [B,8,T] int64 (post-
    _revert_delay_pattern), the vLLM-Omni output waveform, and — per the BF16
    correction in §9 — the ACTUAL runtime dtype at every stage boundary
    (Talker output, fc1/fc2, each acoustic_decoder layer), not an assumption of
    either FP16 or BF16. Checkpoint storage dtype is now known to be BF16
    (confirmed by inspection); what matters for #48 is whether vLLM's runtime
    --dtype float16 forcing on T4 introduces an FP16 intermediate anywhere
    upstream of or inside Code2Wav. Also capture per-layer FP32 activations
    from a CPU FP32 forward pass of the SAME weights as the comparison
    reference — NOT the broken T4 output.

E3  Weight-norm fold verified.
    Host-side FP32 fold of g*v/||v|| for one conv, compared against PyTorch's
    materialized conv.weight. Must match to FP32 precision before any conv parity
    number is meaningful.

E4  ConvTranspose1d GPU-executability answered (§6).
    Does max.nn.ConvTranspose1d run on a T4 GPU, silently fall back to CPU, or error?
    A single tiny graph answers it. Not blocking for correctness, blocking for any
    performance claim.

E5  Divergence detector written before the port.
    Per-layer comparator that reports, against the E2 FP32 reference: max abs err,
    max rel err, NaN/Inf count, EXACT-ZERO count, and saturation count. The zero and
    saturation counters are not optional — M0 showed a NaN/Inf-only scan reports a
    fully-zeroed tensor as healthy, which is the #48 failure shape.
```

### Recommended first prototype: Snake1d

The smallest artifact that retires the most uncertainty per unit of effort is **a MAX graph
computing Snake1d alone, matched numerically against a PyTorch reference.**

```text
input     x     [1, 32, 256] float32, fixed-seed
parameter alpha [1, 32, 1]   float32, including deliberately small values (1e-3, 1e-4)
graph     y = x + (alpha + 1e-9)^-1 * sin(alpha * x)^2
check     max rel err vs torch reference < 1e-6 on CPU and on GPU
then      re-run with x/alpha stored FP16 and the expression cast to FP32 — confirm the
          cast strategy of §9 actually holds, and confirm that WITHOUT it small alpha
          overflows. A demonstrated FP16 failure here is a positive result: it is direct
          evidence for the #48 hypothesis.
```

Why Snake1d rather than a conv:

- It needs **no missing op** — every primitive exists — so a failure is unambiguously a
  *precision* or *broadcast* finding, not "MAX lacks conv1d". It cannot fail for an
  uninteresting reason.
- It is the port's **most numerically dangerous op** (a reciprocal of a learned parameter) and
  appears **37 times** in the stack. If FP16 Snake is what breaks Higgs, this one prototype finds
  it before a single conv is written.
- It exercises the full MAX toolchain end-to-end (graph build → compile → device execute → host
  compare) on both M1/Metal and T4/CUDA, which is a prerequisite for everything else and is worth
  de-risking on the simplest possible graph.
- It directly validates the §9 explicit-cast strategy that every later component depends on.

**Second prototype, immediately after:** one weight-normed dilated Conv1d (`C=32, k=7,
dilation=3, padding=6`) via route A (`ops.conv2d` with a degenerate height axis), weights folded
per E3, compared against PyTorch. That answers the single largest structural question — *can
Higgs's conv shape be expressed in MAX without writing a Mojo kernel?* — and its answer decides
whether M3 is a graph-assembly task or a kernel-authoring task.

## What M1 does not claim

- No MAX code has been written or run. Every "EXISTS" above is an API-surface reading of the
  nightly clone, not a working call.
- The two open questions (§8, §10) are open. Neither was resolved by this pass.
- No statement is made about whether a MAX Code2Wav will actually fix #48. That is precisely the
  falsifiable question M2 exists to answer, and it is set up here so that a negative result is
  as informative as a positive one.
