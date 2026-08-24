# Raw facts: BosonDacDecoder / HiggsAudioRVQ layer stack (haiku extraction, v0.26.0)

Source: `vllm_omni/model_executor/models/higgs_audio_v2/higgs_audio_decoder.py` (471 lines) in
`.research-scratch/vllm-omni` (tag v0.26.0). This is the actual per-layer neural architecture
that `higgs_audio_v3_code2wav.py`'s `self.acoustic_decoder` / `self.quantizer` delegate to.

## Classes

### HiggsAudioVQLayer (36-48)
- `nn.Embedding(codebook_size=1024, codebook_dim=64)` (codebook), `nn.Linear(codebook_dim, hidden_size)` (project_out).
- `decode(indices)`: embedding lookup [B,T]→[B,T,64] → Linear →[B,T,hidden_size] → `.permute(0,2,1)` →[B,hidden_size,T].

### HiggsAudioRVQ (51-77)
- `nn.ModuleList` of 8 `HiggsAudioVQLayer`s.
- `decode(codes)`: allocates `torch.zeros(..., dtype=torch.float32)` (line 73, the ONLY explicit
  dtype in this whole file), then **sums** each quantizer's decode output (residual accumulation
  across the 8 codebooks) — not concatenation.

### _Snake1d (88-99) + `_snake` (80-85, `@torch.jit.script`)
- Per-channel learnable activation: `x + (alpha+1e-9)^-1 * sin(alpha*x)^2`, `alpha` shape
  `[1, channels, 1]`. Reshapes to `[B,C,-1]`, applies, reshapes back.

### _wn_conv1d / _wn_conv_transpose1d (102-111)
- Thin wrappers: `weight_norm(nn.Conv1d(...))` / `weight_norm(nn.ConvTranspose1d(...))`
  (`torch.nn.utils.weight_norm`).

### _BosonResidualUnit (114-130)
- `block = Sequential(Snake1d(dim), wn_conv1d(dim,dim,k=7,dilation=d,padding=(7-1)*d//2),
  Snake1d(dim), wn_conv1d(dim,dim,k=1))`.
- forward: `y=block(x)`; crop `x` symmetrically to `y`'s temporal length if `y` is shorter
  (`pad=(x.shape[-1]-y.shape[-1])//2`, `x=x[...,pad:-pad]`); return `x+y`.

### _BosonDecoderBlock (133-154)
- `block = Sequential(Snake1d(input_dim), wn_conv_transpose1d(input_dim, output_dim,
  kernel_size=2*stride, stride=stride, padding=ceil(stride/2), output_padding=stride%2),
  ResidualUnit(output_dim,dilation=1), ResidualUnit(output_dim,dilation=3),
  ResidualUnit(output_dim,dilation=9))`.

### BosonDacDecoder (157-207) — the actual vocoder
- Defaults: `input_channel=256, channels=1024, rates=(8,5,4,2,3), d_out=1`.
- `wn_conv1d(256, 1024, k=7, padding=3)` → 5× `_BosonDecoderBlock` halving channels each stage
  (1024→512→256→128→64→32, strides 8/5/4/2/3) → `Snake1d(32)` → `wn_conv1d(32, 1, k=7, padding=3)`.
- `hop_length` = product of rates = **960** (matches the v3 wrapper's hardcoded value exactly).
- Pure `nn.Sequential`, no top-level skip connection around the whole stack.

## Build functions
- `build_boson_dac_decoder(device)`: `BosonDacDecoder().to(device)` — all-defaults instantiation,
  no config plumbed through at all (channel counts above are load-bearing hardcoded defaults, not
  read from any Higgs config field).
- `build_higgs_audio_acoustic_decoder(tokenizer_config, device)`: builds a **HuggingFace
  `transformers.DacModel`** (`DacConfig(**tokenizer_config["acoustic_model_config"])`), takes
  `.decoder` submodule, calls `adjust_conv_transpose_output_padding(decoder)` (not traced further),
  and replaces `decoder.tanh` with `nn.Identity()` if present. This is a *second, independent*
  decoder implementation path (HF's own DAC, not the hand-written BosonDacDecoder above) —
  which one actually gets used depends on which factory the v3 wrapper calls, not resolved by
  this extraction pass.

## Dtype
Exactly one explicit dtype anywhere in this file: `torch.float32` for the RVQ's zero-initialized
accumulator (line 73). Every Conv1d/ConvTranspose1d/Embedding/Linear layer here runs at
whatever dtype its weights are loaded in — there is no hardcoded bf16/fp16 anywhere in the real
layer stack, matching the wrapper file's own finding.

## Relevance to MAX port (facts only, no recommendation)

The building blocks a MAX port of this stage needs, in order of appearance:
`nn.Embedding` → `nn.Linear` → (8-way sum) → `weight_norm(Conv1d)` → `Snake1d` (custom activation,
JIT-scripted formula, trivially portable) → 5×(`weight_norm(ConvTranspose1d)` + 3×
`weight_norm(Conv1d)` residual units with dilations 1/3/9 and temporal-crop residual add) →
`Snake1d` → `weight_norm(Conv1d)`. Two decoder implementations exist upstream
(hand-written `BosonDacDecoder` vs. HF `transformers.DacModel`); a MAX responsibility map needs
to know which one the actual v3 checkpoint's `_load_from_bundled_state` path selects before
committing to porting one specific architecture.
