# E1 resolution: real `bosonai/higgs-tts-3-4b` checkpoint inspection

Date: 2026-08-24. Deterministic, reproducible inspection — no MAX/Mojo code involved, no full
weights downloaded. Method: fetch `model.safetensors.index.json` (key list) and the
`model.safetensors` file's own header (first 8 bytes = header length, then that many bytes of
JSON with per-tensor `dtype`/`shape`/`data_offsets`) via HTTP range requests. Reproducible with:

```bash
curl -sL "https://huggingface.co/bosonai/higgs-tts-3-4b/resolve/main/model.safetensors.index.json" -o index.json
curl -sL -r 0-7 "https://huggingface.co/bosonai/higgs-tts-3-4b/resolve/main/model.safetensors" -o header_len.bin
python3 -c "import struct; print(struct.unpack('<Q', open('header_len.bin','rb').read(8))[0])"
curl -sL -r 8-<len+7> "https://huggingface.co/bosonai/higgs-tts-3-4b/resolve/main/model.safetensors" -o header.json
```

## E1: which decoder branch does the real checkpoint take? RESOLVED — HF `DacModel` branch

`is_boson_layout = any(k.startswith("model.") for k in ad_keys)` in
`higgs_audio_v3_code2wav.py:346-371`. The real checkpoint's codec keys under
`tied.embedding.modality_embeddings.0.model.acoustic_decoder.*` are:

```text
acoustic_decoder.conv1.{weight,bias}
acoustic_decoder.conv2.{weight,bias}
acoustic_decoder.snake1.alpha
acoustic_decoder.block.{0..4}.conv_t1.{weight,bias}
acoustic_decoder.block.{0..4}.res_unit{1,2,3}.conv{1,2}.{weight,bias}
acoustic_decoder.block.{0..4}.res_unit{1,2,3}.snake{1,2}.alpha
acoustic_decoder.block.{0..4}.snake1.alpha
```

**None start with `"model."`** → `is_boson_layout = False` → the real checkpoint takes the
`build_higgs_audio_acoustic_decoder` (HF `DacModel`) branch, confirming upstream's source
comment empirically. **§8 of the responsibility map is now CLOSED, not just "asserted."**
`m1-responsibility-map.md`'s §5–§7 weight-name tables (`model.N.*` positional layout) describe
`BosonDacDecoder` and are **not** the real checkpoint's layout — the real key schema is
`block.N.{conv_t1,res_unit1,res_unit2,res_unit3,snake1}.*`, one level of `block.N` nesting with
named submodules, not a flat positional `nn.Sequential`.

## New finding: the RVQ codebook is NOT a plain `nn.Embedding`

`higgs_audio_decoder.py`'s `HiggsAudioVQLayer` (the class `_load_from_bundled_state` builds) has
only `codebook` (`nn.Embedding`) and `project_out` (`nn.Linear`). The real checkpoint's
quantizer keys are richer:

```text
quantizer.quantizers.{0..7}.codebook.embed          [1024, 64]  BF16
quantizer.quantizers.{0..7}.codebook.embed_avg       [1024, 64]  (EMA training buffer)
quantizer.quantizers.{0..7}.codebook.cluster_size    [1024]      (EMA training buffer)
quantizer.quantizers.{0..7}.codebook.inited          scalar      (EMA training buffer)
quantizer.quantizers.{0..7}.project_in.{weight,bias}             (NOT present in the wrapper's class)
quantizer.quantizers.{0..7}.project_out.{weight,bias}
```

This is the standard EMA-updated VQ-VAE codebook parametrization (`embed`/`embed_avg`/
`cluster_size`/`inited` — same shape as e.g. lucidrains' `vector-quantize-pytorch`), used during
*training*; at *inference* only `codebook.embed` (as the effective embedding table) and
`project_out` are needed — `embed_avg`/`cluster_size`/`inited` are training-only EMA state, and
`project_in` is the encoder-side projection (only needed for encoding audio into codes, not for
decoding codes into audio, which is all Code2Wav does). This is consistent with, not
contradicting, the wrapper's simpler `HiggsAudioVQLayer(codebook, project_out)` — the wrapper's
`_load_from_bundled_state` most likely maps `codebook.embed` → `codebook.weight` and drops the
rest, but this pass did not read that remapping code closely enough to confirm the exact key
translation. **Flag for whoever writes the MAX port's weight loader**: don't assume a 1:1 key
rename: `project_in` must be dropped deliberately, not treated as a missing-key error.

## Major finding: the checkpoint's codec weights are stored in BF16, not FP16

Sampled tensor dtypes from the real `model.safetensors` header (not downloaded, not inferred):

```text
acoustic_decoder.block.0.conv_t1.weight              BF16  [1024, 512, 16]
acoustic_decoder.block.4.res_unit3.conv2.weight       BF16  [32, 32, 1]
acoustic_decoder.snake1.alpha                         BF16  [1, 32, 1]
acoustic_decoder.block.0.res_unit1.snake1.alpha       BF16  [1, 512, 1]
fc1.weight / fc2.weight                               BF16  [768,1024] / [256,1024]
quantizer.quantizers.0.codebook.embed                 BF16  [1024, 64]
quantizer.quantizers.0.project_in.weight              BF16  [64, 1024]
quantizer.quantizers.0.project_out.weight             BF16  [1024, 64]
body.layers.0.{input_layernorm,mlp.down_proj,...}     BF16  (LLM backbone, for comparison)
```

Every sampled tensor — codec and LLM backbone alike — is BF16. This is a whole-checkpoint
property, not a codec-specific quirk. `config.json` declares no explicit `torch_dtype` field, so
this is only knowable by reading the actual tensor headers, which is what this pass did.

### Why this matters more than the responsibility map's original framing assumed

`m1-responsibility-map.md` §3/§7/§9 discuss the precision hazard mostly in terms of **FP16**
(matching M0's FP16-focused numerical suite: `matmul fp16`, `rmsnorm fp16`, the FP16 finite range
of 65504). The real checkpoint is BF16, which has a **materially different failure profile**:

```text
              exponent bits   mantissa bits   finite range        failure mode
FP16          5               10              ±65504              OVERFLOW to Inf on large
                                                                    magnitude (M0's finding)
BF16          8               7               same as FP32         no overflow at FP32-like
                                                                    magnitudes; failure mode is
                                                                    PRECISION LOSS (coarse
                                                                    mantissa), not range
```

**Consequence: the specific M0 finding "FP32 accumulation reduces but does not repair FP16
overflow because the true value exceeds FP16's 65504 ceiling" does NOT directly explain a BF16
checkpoint's behavior on T4** — BF16 doesn't have that ceiling. If the true root cause of #48
were value-magnitude overflow the way M0's synthetic FP16 test reproduced, it should not occur
in BF16 storage. Two more-consistent hypotheses to carry into M2, that this pass does not resolve
but which now rank above "FP16-style overflow of a BF16 tensor":

1. **BF16-on-Turing (sm_75) has a genuine correctness gap in PyTorch/cuDNN for these specific
   ops** (weight-normed transposed convs, or the RVQ codebook gather), not just a performance
   gap — T4 has no BF16 tensor cores, so any BF16 kernel either runs via a generic-CUDA-core
   fallback (should be correct, slow) or hits an unsupported-configuration code path that
   produces silently wrong results. This is a *software support* hypothesis, not a *numeric
   range* hypothesis, and it reframes what M2's divergence detector needs to isolate: run the
   same BF16 weights through the same ops on CPU (reference) vs T4 GPU (suspect), not BF16 vs
   FP16.
2. **The `.to(dtype=...)` re-casts inside `decode_codes`** (`quantized.to(dtype=self.fc2.weight.dtype)`,
   `quantized.to(dtype=first_param.dtype)`) could still introduce an FP16 intermediate if
   vLLM-Omni's `--dtype float16` server flag (used on T4 per this project's own
   `configs/higgs_multimodal_qwen3_turing.yaml`, because vLLM refuses bf16 checkpoints on
   pre-Ampere hardware for the LLM stage) forces the **Talker** stage to run/store in FP16, and
   that FP16-ness then propagates into Code2Wav's input even though the codec weights themselves
   stay BF16. This would make the M0 FP16-overflow finding relevant again, but at a different
   point in the pipeline (the Talker's hidden-state/logit path, not the codec weights) than this
   document's §2–§7 assumed. **Not resolved by this pass** — needs the actual dtype of the codes
   tensor and any intermediate hidden state crossing the Talker→Code2Wav boundary on a real T4
   run, which E2 (the reference fixture capture) should record explicitly.

**Action for the responsibility map**: E2 (reference-fixture capture) must record the *actual*
runtime dtype of every tensor at the Talker→Code2Wav boundary and at each Code2Wav internal
stage on the real T4 run that reproduces #48 — not assume either FP16 or BF16 from this
checkpoint-only inspection. This inspection tells us the **weights on disk** are BF16; it does
not tell us what dtype vLLM's `--dtype float16` flag forces at **runtime**, which is the actual
open question for #48.
