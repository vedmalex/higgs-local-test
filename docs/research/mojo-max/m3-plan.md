# M3 plan — one real `_BosonDecoderBlock` as a mixed CPU/GPU MAX graph (issue #57)

Date: 2026-08-24. Milestone M3 of issue #57. **This is a planning document.** Nothing in it has
been executed. No task below is checked off; per repo policy (`AGENTS.md`) a box is ticked only
after the thing has actually been RUN and its real output recorded in a results doc.

Prior milestones: [`m0-results.md`](m0-results.md) (hardware probe),
[`m1-responsibility-map.md`](m1-responsibility-map.md) (component map + E1–E5 entry criteria),
and four M2 prototypes: [`m2-snake1d-results.md`](m2-snake1d-results.md),
[`m2-conv1d-results.md`](m2-conv1d-results.md),
[`m2-convtranspose1d-results.md`](m2-convtranspose1d-results.md),
[`m2-residual-unit-results.md`](m2-residual-unit-results.md).

## Where M2 left things (facts, not re-derived here)

```text
Snake1d (plain MAX ops)          GPU PASSED on M1/Metal AND T4/CUDA — see the "T4 result" section
                                 of m2-snake1d-results.md. FP32 compute safe; FP16-without-cast
                                 breaks hard (1024 NaN/Inf), identically on both platforms.
wn Conv1d via ops.conv2d (A)     GPU PASSED on M1 AND T4 — see the "T4 result" section of
                                 m2-conv1d-results.md. weight_norm folded FP32 on host.
ConvTranspose1d                  CPU PASSED on both (all 5 real (stride,output_padding) pairs).
                                 GPU FAILS on both: Metal = fatal abort (cudnnCreate missing);
                                 T4 = CUDNN_STATUS_ALLOC_FAILED on all 5, uniformly.
_BosonResidualUnit composite     GPU PASSED on M1 (max|err| 4.10e-06). Not re-run on T4.
Real checkpoint                  bosonai/higgs-tts-3-4b, HF DacModel decoder layout, BF16 storage.
```

Nothing prototyped so far touched real checkpoint weights or BF16 storage end-to-end.

---

## 1. Scope decision

**Recommended M3 deliverable: exactly one `_BosonDecoderBlock`, assembled as a single mixed
CPU/GPU MAX graph, numerically validated against an FP64 NumPy reference — with synthetic
weights first, then with the real BF16 checkpoint weights for that one block.**

Concretely the block is (`higgs_audio_decoder.py:133-154`, HF `DacModel` naming per §8 of the map):

```text
_BosonDecoderBlock(input_dim, output_dim, stride)
    Snake1d(input_dim)                                   GPU  (M2-confirmed)
    wn_conv_transpose1d(input_dim, output_dim,
        k=2*stride, stride=stride,
        padding=ceil(stride/2), output_padding=stride%2)  CPU  (GPU broken upstream)
    ResidualUnit(output_dim, dilation=1)                  GPU  (M2-confirmed composite)
    ResidualUnit(output_dim, dilation=3)                  GPU
    ResidualUnit(output_dim, dilation=9)                  GPU
```

Pick **`stride=5`** as the primary case (`DecoderBlock(512, 256, stride=5)`): stride-5 and
stride-3 both have `output_padding=1` (stride-5 is not unique on that property), but stride-5 is
the block with the largest channel change of the two (`512→256` vs stride-3's smaller ratio) and
sits earlier in the decoder's upsampling chain, so it exercises both the off-by-one
output-length hazard §6 of the map flags as the compounding risk *and* a non-trivial channel
resize in the same case. Add `stride=8` (`1024→512`, `output_padding=0`) as a second case for
shape coverage.

**Why not smaller.** Every sub-piece of this block except `ConvTranspose1d` is already validated
in isolation *and* as a composite. Another isolated-op prototype retires no uncertainty. The two
things still genuinely unknown are (a) can one MAX graph legally mix a CPU-placed op into an
otherwise-GPU graph, and (b) does the crop/output-length arithmetic survive a real transposed
conv followed by three residual units. Only a whole block tests both.

**Why not larger.** The full 5-block `BosonDacDecoder` (37 convs, 37 Snakes) multiplies the
device-transfer count by 5 and, per §7, compounds error across depth — debugging a divergence
there without a validated single block first is strictly worse. Full-decoder assembly is M4.

**Against the map's entry criteria.** The map states M2 entry criteria E1–E5 but no explicit
M2→M3 criteria. Read against E1–E5, M3 inherits three that are still **not** done and folds them
in as tasks below rather than pretending they are prerequisites already met:

```text
E1  DONE (m1-facts-checkpoint-inspection.md) — HF DacModel layout confirmed.
E2  NOT DONE — no real fixture captured, no runtime dtypes recorded on Tesla T4.
    M3 does the WEIGHT half of E2 (real block weights) but NOT the broken-T4-run half.
E3  PARTIAL — the fold is implemented in all M2 prototypes and matches an FP64 reference,
    but has never been checked against PyTorch's materialized conv.weight on real weights.
    M3-3 below closes this.
E4  DONE (answered NO for GPU, YES for CPU) — m2-convtranspose1d-results.md.
E5  NOT DONE — no reusable per-layer divergence detector exists; each prototype
    hand-rolls its own max|err| print. M3-2 below builds the real one.
```

---

## 2. Staged task list

Ordering matters: **M3-1 is a blocking spike, but only for Stage C onward.** If M3-1 says
mixed-device single-graph does not work, M3-5 onward (the tasks that actually assemble a MAX
graph) change shape (two graphs stitched in Python) and the plan must be re-reviewed before that
part of execution continues. **Stage B (M3-3, M3-4 — real-weight extraction and the FP64/FP32
reference implementation) is explicitly NOT gated by M3-1**: both are pure host-CPU
NumPy/PyTorch work with no MAX graph device placement involved, so they can run in parallel with,
or entirely independently of, the M3-1 spike. Their "depends on" is nothing upstream of Stage A's
M3-2 detector (which they use), not M3-1.

### Stage A — resolve the blocker and build the tooling

- [x] **M3-1 (SPIKE, BLOCKING). Prove or disprove mixed CPU/GPU placement inside one MAX graph.**
  **DONE — PASSED.** Ran on the pinned M1 toolchain, isolated subprocess, 3 clean exits (code 0),
  output matching an inline FP64 reference (max|err|=3.48e-07, max_rel_err=1.68e-05, no NaN/Inf).
  Mixed-device single-graph placement works; the two-graph fallback is NOT required. Found and
  worked around an unrelated upstream layout bug (`bias=` into `ops.conv2d_transpose` with H=1
  NHWC input mis-shapes the output) — added to the M3-11 bug-report candidate list, not conflated
  with this verdict. Detail: [`m3-device-mixing-results.md`](m3-device-mixing-results.md).
  Minimal graph: GPU Snake1d → `ops.transfer_to(CPU)` → `ops.conv2d_transpose` on CPU-placed
  operands → `ops.transfer_to(GPU)` → GPU Snake1d. Session constructed with
  `InferenceSession(devices=[Accelerator()])` (host CPU is appended automatically —
  `max/python/max/engine/api.py:587`). Not just the activation tensor: the ConvTranspose1d's
  filter/bias constants must also be placed on a CPU `DeviceRef` before the op — a CPU-placed
  activation with a GPU-placed weight/bias would not actually test (or achieve) CPU dispatch of
  the op.
  *Pinned toolchain this plan targets* (this proof is version-specific, not a general MAX
  guarantee): MAX 26.5.0, Mojo 1.0.0 (`ed45d567`), on M1: macOS 26.6.2 (25G83) + Xcode 26.6
  (24959, build 17F113) + Metal compiler 32023.883; on T4: Mojo 1.0.0 / MAX 26.5.0 via
  `notebooks/mojo_max_m0_t4.ipynb`'s Colab CUDA runtime (per `m0-results.md`, no more specific
  CUDA driver/toolkit version is recorded there than "Colab's default Tesla T4 runtime" — if a
  more precise CUDA driver version is needed later, it must be captured fresh, not assumed).
  *Done when:* the graph compiles and executes on M1 and prints a finite output matching an FP64
  NumPy reference, **and** it is demonstrated that the transposed conv really ran on CPU (i.e. the
  Metal `cudnnCreate` abort did NOT occur — the abort is itself the negative signal, so a clean
  run is proof of CPU dispatch **on this pinned toolchain**). If it aborts: record that, and
  M3-1's deliverable becomes the two-graph fallback measurement instead.
  *Caveat on re-use of this criterion:* "a clean run without crashing" is only a valid proof of
  CPU dispatch as long as `ops.conv2d_transpose`'s GPU path keeps failing the way
  `m2-convtranspose1d-results.md` documents. If this spike (or a later re-check) is re-run after
  an upstream MAX fix to `conv2d_transpose`'s GPU path, a clean run would no longer distinguish
  "ran on CPU as intended" from "ran on GPU because the bug got fixed" — at that point a positive
  placement check is required instead of the absence-of-crash proxy, e.g. inspecting the
  compiled graph's per-op device assignment, or a `MODULAR_DEBUG`-style device-inspection env var
  if MAX exposes one. Add that positive check before relying on this criterion again post-fix.
  *Files:* new `docs/research/mojo-max/m3_device_mixing_spike.py`,
  new `docs/research/mojo-max/m3-device-mixing-results.md`.
  *Devices:* M1 CPU + M1 Metal GPU in one session.
  *Tier:* **sonnet-deterministic-code** (small, mechanical, but the abort-handling needs
  subprocess isolation exactly as `m2_convtranspose1d_prototype.py` needed it).

- [x] **M3-2. Write the reusable per-layer divergence detector (closes E5).**
  **DONE.** `m3_divergence.py`'s `compare()` reproduced the published residual-unit number
  exactly (4.09571e-06 / 9.91201e-04, rounding to the published 4.10e-06 / 9.91e-04), plus 3
  self-check assertions (normal case, all-zero-vs-nonzero #48 shape, FP16-saturation case) all
  pass. Detail: [`m3-divergence-results.md`](m3-divergence-results.md).
  One module, importable by every M3 script: given `got` and an FP64 `ref`, report max abs err,
  max rel err, NaN/Inf count, **exact-zero count**, and saturation count (per §11 E5 — a
  NaN-only scan passes a fully-zeroed tensor, which is the #48 failure shape).
  *Done when:* run standalone on the existing `m2_residual_unit_prototype.py` inputs, it
  reproduces the already-published `max|err|=4.10e-06` from `m2-residual-unit-results.md` — i.e.
  it is validated against a known number, not just written. **This task's pass/fail is decoupled
  from M3-1**: it is validated solely against that already-published residual-unit number,
  independent of whether the M3-1 spike passes, aborts, or falls back to the two-graph design. It
  is additionally *reused by* the M3-1 spike script once written, but that reuse is not this
  task's completion criterion.
  *Files:* new `docs/research/mojo-max/m3_divergence.py`.
  *Devices:* host only.
  *Tier:* **sonnet-deterministic-code**.

### Stage B — real weights for one block

- [x] **M3-3. Extract one real decoder block's weights from the checkpoint and verify the
  weight-norm fold against PyTorch (closes E3 properly).**
  **DONE — with a premise correction.** The real checkpoint has NO `weight_g`/`weight_v` split
  anywhere under `acoustic_decoder.*` (confirmed by grepping all 927 tensor keys) — weight_norm
  was already folded upstream before this checkpoint was written; `build_higgs_audio_acoustic_
  decoder()` never calls `apply_weight_norm()`. Used block 1 (`conv_t1.weight` shape
  `[512,256,10]`, confirming stride=5, 512→256). Verified via PyTorch's own `right_inverse`
  weight_norm decomposition + FP32 fold vs PyTorch's FP32-materialized `conv.weight` across all 7
  kernels: max abs err **1.192e-07** (< 1e-6 gate, PASS). Alpha distribution: 2048 values across 7
  Snake layers; smallest real `|alpha|` ≈ 0.0033, four orders of magnitude from the 1e-7 dangerous
  regime — **no real alpha crosses it**. Secondary finding: many real alphas are negative (up to
  74/256 channels in one layer), worth noting for future synthetic fixtures. Detail:
  [`m3-block-results.md`](m3-block-results.md).
  Pull `acoustic_decoder.block.N.{conv_t1, res_unit1..3.{conv1,conv2}, snake1.alpha}` (+ the
  residual units' own `snake{1,2}.alpha`) for the stride-5 block from `bosonai/higgs-tts-3-4b`.
  **Precision sequencing, made explicit to avoid an incoherent comparison:** the checkpoint's raw
  `g`/`v` tensors are stored BF16; both the weight-norm fold (`g*v/||v||`) AND the comparison
  against PyTorch's materialized `conv.weight` happen in FP32 — i.e. `g` and `v` are upcast from
  BF16 to FP32 (or FP64, matching this task's other host-side folds) *before* folding and *before*
  comparing. The BF16 downcast of the folded result happens strictly AFTER this FP32 comparison
  passes, not before. (BF16's epsilon is ~7.8e-3; comparing a BF16-rounded fold against an FP32
  PyTorch reference at a < 1e-6 abs-err band would be mathematically impossible to pass and is
  not what is being tested here — this task tests the fold arithmetic, not a BF16 storage
  round-trip.)
  *Done when:* every FP32-folded kernel matches PyTorch's FP32-materialized `conv.weight` to
  < 1e-6 max abs err, and the real `alpha` value distribution is printed (this is the open
  question from `m2-snake1d-results.md`: is any trained `alpha` small enough to reach the
  dangerous reciprocal regime at all? — per that doc's finding, `alpha` at or below ~1e-7 is the
  regime that triggers the FP16 `1/alpha` overflow, so report explicitly whether any real
  `alpha` falls at or below that threshold).
  *Files:* new `docs/research/mojo-max/m3_block_weights.py`, results appended to a new
  `docs/research/mojo-max/m3-block-results.md`.
  *Devices:* host CPU only (torch reference).
  *Tier:* **sonnet-deterministic-code**, with **opus** review of the alpha-distribution
  conclusion (that is an interpretive claim about #48, not a mechanical number).

- [x] **M3-4. Build the FP32 reference implementation of the whole block in NumPy FP64.**
  **DONE.** New FP64 `numpy_conv_transpose1d` validated against `torch.nn.functional.
  conv_transpose1d` to 1.11e-16. Full block cross-checked three ways: synthetic weights stride-5
  (max|err|=2.25e-06, PASS), synthetic stride-8 (max|err|=2.38e-06, PASS), and real checkpoint
  weights stride-5 via M3-3's extraction (max|err|=1.27e-05 — misses the literal < 1e-5 gate by a
  small margin). Diagnosed rather than just reported: a torch FP32-vs-FP64 control run on the
  identical chain showed the exact same 1.27e-05 gap, while this NumPy-FP64 reference agrees with
  torch's own FP64 forward to 2.18e-14 — the gap is FP32 rounding-order noise at real-checkpoint
  depth/magnitude, not a reference bug, and is well inside M3-5/M3-6's actual gating metric (max
  relative error ≤ 5e-03). FP64 path remains licensed as ground truth. Detail:
  [`m3-block-results.md`](m3-block-results.md).
  Extend `numpy_residual_unit` (already in `m2_residual_unit_prototype.py`) with a FP64
  `numpy_conv_transpose1d` and the block wiring, cross-checked against a PyTorch FP32 forward of
  the real `DacModel` block for the same inputs.
  *Done when:* NumPy-FP64 and PyTorch-FP32 agree to < 1e-5 max abs err on real weights, so the
  FP64 path is licensed as the reference for MAX comparison.
  *Files:* `docs/research/mojo-max/m3_block_reference.py` (new).
  *Devices:* host CPU.
  *Tier:* **sonnet-deterministic-code**.

### Stage C — the block in MAX

- [ ] **M3-5. Assemble the full `_BosonDecoderBlock` as one MAX graph, FP32 throughout, synthetic
  weights, stride=5.** Reuses `snake_expr` and `conv1d_expr` verbatim from
  `m2_residual_unit_prototype.py`; adds the route-A `conv_transpose_expr` from
  `m2_convtranspose1d_prototype.py`, CPU-placed per M3-1's answer.
  *Done when:* it runs on M1 (GPU for everything but the transposed conv) and the M3-2 detector
  reports, against the M3-4 FP64 reference: **max relative error ≤ 5e-03 as the primary gating
  metric** — this band is derived directly from M2's own measured full-composite numbers on real
  hardware, not guessed: conv1d T4 max rel err 6.45e-04 and residual-unit (M1) max rel err
  9.91e-04 (see `m2-conv1d-results.md` and `m2-residual-unit-results.md`); a full block chains a
  transposed conv and three residual units, so 5e-03 gives roughly 5x headroom over the largest
  measured single-composite relative error, generous but not slack. Max abs err (cf. 2.37e-06 /
  4.10e-06 in the M2 docs) is still recorded and reported, but only as a secondary/informational
  metric — an absolute-error band is not a coherent gate across tensors of varying magnitude at
  block-composite depth. Also required: zero NaN/Inf, zero unexplained exact zeros, **and the
  output length matches the reference exactly** (no ±1).
  *Files:* new `docs/research/mojo-max/m3_decoder_block_prototype.py`.
  *Devices:* M1 Metal GPU + CPU, one session.
  *Tier:* **sonnet-deterministic-code**.

- [ ] **M3-6. Re-run M3-5 with the real stride-5 checkpoint weights (from M3-3).**
  *Done when:* same tolerances as M3-5 (max relative error ≤ 5e-03 primary gate, max abs err
  secondary/reported-only) against the M3-4 real-weight reference, on real weights.
  Divergence here with M3-5 passing localizes the problem to weights/layout, not graph
  structure — which is exactly the "wrong layout vs wrong precision" discrimination §6 demands.
  *Files:* same script, `--real-weights` path; results in `m3-block-results.md`.
  *Devices:* M1 GPU + CPU.
  *Tier:* **sonnet-deterministic-code**.

- [ ] **M3-7. Add the stride=8 block case (`1024→512`, `output_padding=0`).**
  *Done when:* passes the same checks as M3-5/M3-6 (max relative error ≤ 5e-03 primary gate),
  confirming the block builder is stride-generic rather than tuned to one case.
  **Explicitly out of scope for this task and for M3 as a whole:** the stride-5 and stride-8
  blocks are tested independently here, each fed its own synthetic/real input, not chained
  output-of-one-into-input-of-the-other as the real 5-block decoder does. Cross-block
  composability — the device-transfer count across chained blocks, and whether the ±1-sample
  length-mismatch hazard §6/§7 warn about compounds across multiple chained blocks — remains
  untested until a future M4 that assembles the full `BosonDacDecoder`. This is not silently
  assumed to be fine; it is an explicit non-goal here (consistent with §6's "NOT the full 5-block
  `BosonDacDecoder`").
  *Files:* same script.
  *Devices:* M1 GPU + CPU.
  *Tier:* **haiku-research** is NOT appropriate; **sonnet-deterministic-code** (small but it is a
  correctness result).

- [ ] **M3-8. Exercise the `pad > 0` crop branch.**
  `m2-residual-unit-results.md` explicitly records that only the `pad == 0` no-op branch has ever
  run, and — per that same doc — in the real config the dilated conv's `(k-1)*dilation//2`
  padding formula already preserves length exactly, which may make `pad > 0` unreachable for any
  real Higgs decoder-block configuration. **First check reachability**: enumerate whether any
  real `(kernel, dilation, stride)` combination actually used by `_BosonDecoderBlock` produces
  `pad > 0`. If none do, this task is **defensive/edge-case-only** (a synthetic config
  constructed solely to exercise the branch), not a required real-config test, and should be
  labeled as such in the results doc rather than presented as validating real behavior.
  Construct a padding configuration where `len(x) > len(y)` and check the asymmetric
  integer-division behaviour §7 warns about.
  *Done when:* a `pad > 0` case runs and matches the FP64 reference, including the odd-difference
  case where PyTorch's `//` makes the crop asymmetric; the reachability finding (real vs
  synthetic-only) is stated explicitly.
  *Files:* `m3_decoder_block_prototype.py`.
  *Devices:* CPU is sufficient.
  *Tier:* **sonnet-deterministic-code**.

### Stage D — precision

- [ ] **M3-9. BF16-storage / FP32-compute pass over the whole block.**
  Per §5 below. Three variants, same block, same inputs: all-FP32 (baseline, from M3-6);
  BF16 storage + explicit FP32 compute; BF16 storage + BF16 compute (no cast).
  *Done when:* all three run on M1 and the detector's numbers for each are recorded against
  **pre-declared numeric thresholds** that separate the three outcomes, rather than a qualitative
  call:
  - **fine**: max relative error ≤ 5e-03 (the same M3-5/M3-6 gating band), zero NaN/Inf, zero
    unexplained exact zeros.
  - **degrades**: max relative error > 5e-03 but ≤ 5e-02 (an order of magnitude beyond the gating
    band), still zero NaN/Inf and zero unexplained exact zeros — usable but worse than the FP32
    baseline, not a correctness failure.
  - **breaks**: any NaN/Inf, any unexplained exact-zero tensor, or max relative error > 5e-02.
  Additionally, report the real `alpha` values (from M3-3) against the **dangerous regime**
  established by `m2-snake1d-results.md`: `alpha` at or below **1e-7** is the regime empirically
  shown to trigger the FP16 `1/alpha` overflow (and by the same reciprocal-overflow logic is the
  threshold to watch under BF16-without-cast too); explicitly state whether any real per-channel
  `alpha` for this block falls at or below that threshold. Per §10 of the map, any BF16-is-safe
  conclusion must be scoped to Metal until M3-10 confirms it on T4.
  *Files:* `m3_decoder_block_prototype.py`, `m3-block-results.md`.
  *Devices:* M1 GPU + CPU.
  *Tier:* **opus-fable-analysis** for interpreting the result (this is the #48-adjacent precision
  claim; do not let a mechanical tier draw the conclusion), sonnet for the code.

### Stage E — Tesla T4 re-validation

- [ ] **M3-10. Re-run M3-1, M3-5/M3-6, M3-7 and M3-9 unchanged on a Colab Tesla T4.**
  Reuse the `notebooks/mojo_max_m2_t4.ipynb` Drive-workspace pattern; a **new** notebook
  `notebooks/mojo_max_m3_t4.ipynb` is needed because the M2 notebook hardcodes the four M2
  prototype scripts. The M1-only `_BosonResidualUnit` T4 gap
  (`m2-residual-unit-results.md`) is closed for free by M3-5 running there.
  *Done when:* the Tesla T4 run's raw output is committed as
  `docs/research/mojo-max/m3-block-output-t4.txt` and every M1 result above is confirmed or
  contradicted per-case in `m3-block-results.md`. Expect the CPU-placed transposed conv to work
  (Tesla T4 CPU already 5/5) — the real risk is whether mixed placement behaves the same under
  CUDA.
  *Files:* new `notebooks/mojo_max_m3_t4.ipynb`, new `m3-block-output-t4.txt`.
  *Devices:* Colab Tesla T4 GPU + CPU.
  *Tier:* **sonnet-deterministic-code** (notebook is mechanical), **opus** for reconciling any
  M1↔Tesla-T4 disagreement.

### Stage F — upstream (does not gate anything above)

- [ ] **M3-11. File the `conv2d_transpose` GPU bug against `modular/modular`.**
  Not a dependency of any task above; M3's whole device-mixing design exists to route around it.
  Exact script to reference in the report: `m2_convtranspose1d_prototype.py` (plus the isolated
  per-case subprocess runner it needed, per `m2-convtranspose1d-results.md`'s "Why this needed
  subprocess isolation" section). Minimal repro to describe: the five real Higgs
  `(stride, output_padding)` pairs `(8,0) (5,1) (4,0) (2,0) (3,1)` with `k=2*stride`, 16 channels,
  built via `ops.conv2d_transpose` route-A layout, executed once on `CPU()` and once on
  `Accelerator()`. Two distinct failure modes to report as one issue with two sections: (i)
  **Metal** — fatal process abort, `symbol not found: cudnnCreate`, i.e. the GPU dispatch attempts
  a CUDA-only library load on a non-CUDA accelerator, uncatchable by `try`/`except`; (ii)
  **Tesla T4 / sm_75** — catchable `CUDNN_STATUS_ALLOC_FAILED` at `conv_transpose_mogg` on **all
  five** cases, kernel sizes 4 through 16 identically, on a fresh 15 GB T4 with tiny tensors — the
  uniformity across kernel size rules out a shape-specific workspace-sizing bug and points at the
  dispatch path itself. Note upstream's own `# TODO(GEX-2043)` and that CPU is correct on both
  platforms. Include the same toolchain version-pinning block used in M3-1 (MAX 26.5.0, Mojo
  1.0.0, macOS 26.6.2 + Xcode 26.6 for M1; Mojo 1.0.0 / MAX 26.5.0 on the Colab Tesla T4 runtime)
  so the report is reproducible against a specific version rather than "MAX" generically.
  *Done when:* the issue exists and its URL is recorded in `m2-convtranspose1d-results.md` and on
  issue #57.
  *Files:* one-line link added to `m2-convtranspose1d-results.md`.
  *Devices:* none.
  *Tier:* **opus-fable-analysis** — this is user-facing external writing, and the two-failure-mode
  framing is the whole value of the report.

---

## 3. Device-mixing mechanics

**What the API actually provides** (read from `.research-scratch/modular`, not from memory):

```text
DeviceRef                      max/python/max/graph/type.py — every TensorType carries a device;
                               DeviceRef.CPU() / DeviceRef.from_device(obj)
ops.transfer_to(x, device)     max/python/max/graph/ops/transfer_to.py — inserts an mo.transfer
                               node INTO the compiled graph. Docstring is explicit that this is
                               for "host-to-device input staging, device-to-host output
                               retrieval, or cross-GPU tensor movement inside forward()".
                               No-op if src == dst.
TensorValue.to(device)         max/python/max/graph/value.py:788 — same thing, method form.
InferenceSession(devices=[..]) max/python/max/engine/api.py:557-599 — takes a LIST, and the host
                               CPU is ALWAYS appended if not present (line 587).
precedent                      max/python/max/pipelines/architectures/autoencoders/vae.py:72 and
                               autoencoders_modulev3/autoencoder_kl_flux2.py:167 both call
                               ops.transfer_to inside forward, and flux2 declares CPU-device
                               TensorTypes alongside GPU ones in ONE graph.
```

So the reading of the source is: **the source suggests MAX's `Graph`/`ops` API supports a single
mixed-device graph** (the `DeviceRef`-per-`TensorType` design, `ops.transfer_to`'s explicit
docstring support for this use case, and the `vae.py`/`flux2` precedent of declaring CPU- and
GPU-device `TensorType`s inside one forward pass) — **but this is unverified until the M3-1 spike
actually runs it.** No two-graph stitching has been shown to be *necessary*, but it has also not
yet been shown to be *unnecessary*: that is exactly the empirical gap M3-1 exists to close. The
intended M3 shape, pending that verification, is one graph whose Snake/Conv1d tensors are
GPU-placed and whose transposed-conv operands are `transfer_to(DeviceRef.CPU())`'d immediately
before the op and transferred back immediately after.

**Why M3-1 is still a blocking spike and not a settled design.** Nothing above proves that
`ops.conv2d_transpose` *dispatches to the CPU kernel* purely because its operands are CPU-placed
while the session hosts an accelerator. The Metal failure mode is a fatal abort inside the GPU
dispatch path, so if placement is ignored the spike dies with a process abort rather than a wrong
number — an unambiguous signal either way. Do not write M3-5–M3-9 before M3-1 has run.

Fallback if M3-1 fails: two graphs (GPU prefix, CPU transposed conv, GPU suffix) in one Python
driver, with `Buffer.to(device)` hops between `model.execute` calls — correctness-equivalent,
more host round-trips, and it makes the eventual performance story worse in a way worth
documenting.

**Relation to the Mojo skills.** `mojo-syntax` and `mojo-gpu-fundamentals` govern hand-written
Mojo kernels (`DeviceContext`, `enqueue_create_buffer`, `map_to_host`, `has_accelerator`).
**M3 writes no Mojo kernel** — route A (`ops.conv2d`) already removed that need per
`m2-conv1d-results.md`, and MAX graphs are built in Python. Those skills become binding only if
M4 needs route B (a custom Mojo conv1d) or a fused Snake kernel. Nothing in this plan contradicts
them; it operates one layer above.

---

## 4. Precision handling for the full block

The checkpoint stores **BF16**, not FP16 (§9 of the map). BF16 has FP32's exponent range, so
M2's FP16 Snake blow-up (`1/alpha` past 65504) does **not** transfer as a prediction — but BF16's
8-bit mantissa is far coarser than FP16's, and the map's §10 warns that a BF16 PASS on T4
(`sm_75`, no BF16 tensor cores) may not be hardware BF16 at all.

Policy for M3, unchanged in spirit from §9, extended to the whole block:

```text
1. Storage       weights loaded as BF16 exactly as the checkpoint stores them (no host upcast),
                 so the graph sees the real dtype.
2. Compute       every op's inputs explicitly ops.cast(..., DType.float32) at the block
                 boundary; the entire block computes in FP32; one cast back to BF16 only at the
                 block output (the real decoder chains blocks, so the inter-block hand-off is
                 where storage dtype belongs).
3. Never implicit  MAX's promotion rule never widens beyond an existing operand
                 (graph/dtype_promotion.py). There is no compute_dtype flag. Every widening in
                 M3 must be a visible ops.cast in the source.
4. Fold in FP32  weight_norm's g*v/||v|| stays a host-side FP32 (M3-3 does it in FP64 then
                 downcasts) computation — never a BF16 runtime reduction. Both the fold itself
                 and its comparison against PyTorch's materialized conv.weight happen in FP32;
                 the BF16 downcast is the LAST step, after that comparison passes (see M3-3).
5. Baseline first  M3-5 through M3-8 are all-FP32 end-to-end. BF16 is introduced only in M3-9,
                 one variant at a time, so a divergence has one candidate cause.
6. Scope claims  Any "BF16 is safe" conclusion from M1/Metal is Metal-only until M3-10 confirms
                 it on Tesla T4, per §10's unresolved question about BF16-on-Turing.
```

---

## 5. Verification method

Same rigor as M2, and the same reference ladder:

```text
ground truth     NumPy FP64 (M3-4), itself cross-validated against a PyTorch FP32 forward of the
                 real HF DacModel block to < 1e-5 (so the FP64 path is licensed, not assumed —
                 M2's prototypes used FP64 NumPy as a torch stand-in without this check)
comparator       the M3-2 detector: max abs err, max rel err, NaN/Inf count, EXACT-ZERO count,
                 saturation count
pass band        max RELATIVE error <= 5e-03 is the PRIMARY gating metric for an FP32 block on
                 real weights (M3-5/M3-6/M3-7). Justification: M2's own measured relative errors
                 on real hardware are conv1d (T4) max_rel_err 6.45e-04 and residual-unit (M1)
                 max_rel_err 9.91e-04 — see m2-conv1d-results.md and m2-residual-unit-results.md.
                 A full block chains a transposed conv and three residual units on top of those
                 already-measured composites, so 5e-03 gives roughly 5x headroom over the larger
                 of the two measured numbers: generous but not slack. Max abs err (M2: 2.37e-06
                 conv1d, 4.10e-06 residual unit) is still recorded per-run but is SECONDARY/
                 reported-only, not a gate — an absolute-error threshold does not scale correctly
                 across tensors of different magnitude at block-composite depth, which is why
                 relative error is the gate. A relative-error result WORSE than 5e-03 is a
                 finding to investigate, not a threshold to relax.
shape check      output length must match the reference EXACTLY. A +-1 sample is a FAIL, not a
                 rounding note — it compounds across five blocks (§6) and
                 adjust_conv_transpose_output_padding exists upstream precisely for this.
per-layer        record the detector's numbers at every intermediate tensor (after snake1, after
                 conv_t1, after each residual unit), not only at the block output — §7's
                 explicit instruction: "compare layer-by-layer, not just end-to-end."
honesty          per AGENTS.md, a partial or failing stage is written up as a partial or failing
                 stage. Branch per issue, `Refs #57` on commits, results docs land next to the
                 scripts.
```

---

## 6. Explicit non-goals for M3

- **NOT** the full 5-block `BosonDacDecoder` (37 convs). One block, two stride cases. M4.
- **NOT** the VQ layer, the 8-way RVQ sum, or `fc2`. §1–§3 of the map call these low-risk and
  they retire almost no uncertainty; they are cheap to add once a block works.
- **NOT** end-to-end audio. No waveform is produced, nothing is listened to, no `.wav` is written.
- **NOT** any performance, throughput, or speed comparison — a CPU-placed transposed conv inside
  a GPU graph is a known bottleneck by construction, so any number measured now would be a
  measurement of the upstream bug (M3-11), not of MAX.
- **NOT** the broken-T4-run half of E2: no capture of `audio_codes` or per-stage runtime dtypes
  from a real #48-reproducing vLLM-Omni run. That is a separate, parallel piece of work and M3
  does not depend on it.
- **NOT** a custom Mojo kernel (route B), fused Snake, or any kernel authoring.
- **NOT** waiting for or depending on a Modular fix. M3-11 is filed and then ignored for M3's
  purposes.
- **NOT** a claim about whether MAX fixes #48. M3 establishes that a block can be ported
  correctly; it does not and cannot answer the parity question.

## What this plan does not claim

- M3-1's answer is genuinely unknown. The source reading suggests mixed-device graphs are
  supported and upstream pipelines use them; whether a CPU-placed `conv2d_transpose` avoids the
  GPU dispatch path in a GPU-hosting session has not been executed. Stage C's shape depends on it.
- No estimate is offered for whether the real checkpoint's `alpha` values are in the dangerous
  regime. M3-3 measures it; this plan does not guess.
- The 5e-03 max-relative-error pass band is extrapolated from M2's two measured composites
  (conv1d T4 6.45e-04, residual-unit 9.91e-04), not derived from first principles. If a real
  block lands at, say, 2e-02, the right response is to find out why, not to move the line.
