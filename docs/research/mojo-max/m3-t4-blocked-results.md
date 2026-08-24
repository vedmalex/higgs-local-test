# M3-10 — BLOCKED (upstream): full `_BosonDecoderBlock` fatally aborts on Colab Tesla T4

Date: 2026-08-25. Milestone M3, task M3-10 of issue #57 ("re-run M3-1, M3-5/M3-6, M3-7, M3-9
unchanged on a Colab Tesla T4"). This document supersedes the "M3-10 follow-up" section of
[`m3-block-results.md`](m3-block-results.md) as the final record for this task: the channel-count
hypothesis recorded there was refuted, a further op-level root-cause bisection was proposed as the
next step, and the project owner decided **not** to run that bisection — the cause is upstream,
confirmed via web research against Modular's own forum, issue tracker, and pull request, and no
op-level bisection on our side can produce a fix. This document records that decision and the
evidence behind it.

**Bottom line: Tesla T4 (Turing / `sm_75`) is not a usable platform for this graph on MAX 26.5.0.
The failure is an upstream MAX/Mojo backend gap for Turing GPUs, not a defect in this project's
code, its graph construction, or its numerics. Further T4-side verification for M3 is stopped by
project-owner decision, in favor of continued work on the platforms MAX actually supports.**

---

## 1. What was run

`docs/research/mojo-max/m3_decoder_block_prototype.py` — the same script `m3-block-results.md`'s
M3-5/M3-6/M3-7/M3-9 sections already validated end-to-end on M1/Metal — run unchanged on a fresh
Colab Tesla T4 runtime, via the M3 T4 notebook (`notebooks/mojo_max_m3_t4.ipynb`), same
subprocess-isolation convention this project uses for every fatal-abort-prone run. `DecoderBlock(
512, 256, stride=5)`, synthetic weights, FP32 throughout, `seq_len=20` — the exact M3-5 setup.
Run across all 6 seeds this project's M3-5/M3-6/M3-7 sweeps standardized on: `57305, 1, 2, 3, 42,
12345`.

### Result: fatal abort on all 6 seeds, identically

Every one of the 6 seeds aborted at the same point — after the FP64 NumPy reference has already
been computed and the MAX session has already selected per-stage device placement (GPU for the
Snake/residual-unit stages, CPU for `conv_t1`, exactly as M3-1 established) — during MAX graph
compilation/execution, with:

```text
LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16
```

Process exit code -6 (SIGABRT) in every case. This is a fatal, uncatchable process abort, not a
Python exception — consistent with every other fatal-abort case this project has hit and isolated
via subprocess (`m2-convtranspose1d-results.md`'s Metal `cudnnCreate` abort, this same family of
failure).

**Verbatim output, seed=57305** (the other 5 seeds produce byte-identical output apart from the
`seed=<N>` line and are not reproduced in full here — see
[`m3-block-output-t4.txt`](m3-block-output-t4.txt) for all 6, verbatim):

```text
=== M3-5 stride=5 synthetic seed=57305 ===
--- subprocess stdout ---
accelerator_count=1
stride=5 synthetic weight seed=57305
FP64 reference after_snake1: shape=(1, 512, 20)
FP64 reference after_conv_t1: shape=(1, 256, 100)
FP64 reference after_res_unit1: shape=(1, 256, 100)
FP64 reference after_res_unit2: shape=(1, 256, 100)
FP64 reference after_res_unit3_final: shape=(1, 256, 100)
Session devices requested=[Accelerator()] (host CPU auto-appended); GPU stages device=gpu:0, conv_t1 device=cpu:0
precision=fp32

--- subprocess stderr ---
LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16

--- subprocess exit code: -6 ---
VERDICT: subprocess was killed/aborted (fatal process abort).
```

**Seeds `1, 2, 3, 42, 12345`: identical** — same stdout up through device placement (only the
`stride=5 synthetic weight seed=<seed>` line differs), same stderr line, same exit code -6, same
verdict. The abort happens during graph compilation, downstream of any seed-dependent weight or
input value — consistent with a structural/codegen issue rather than a numerics-triggered one.

---

## 2. What was tried before concluding this is upstream

### 2a. Channel-count kernel-selection hypothesis — tested, REFUTED

The leading hypothesis when the abort was first hit: MAX's `ops.conv2d` kernel-selection heuristic
switches to a tensor-core/GEMM-based kernel above some channel-count threshold, and that kernel
path has a Turing (`sm_75`) codegen gap for `ldmatrix` (an Ampere+-era warp-level matrix-load
instruction).

`m3_ldmatrix_channel_sweep.py` swept `ops.conv2d` (M2's exact, unchanged code path) across
`channels ∈ {32, 64, 128, 256, 512, 1024}` on the same T4, each in its own subprocess. **Every
case PASSED, no abort at any size** — including 1024, 4x the real block's largest conv width
(256) — with max\|err\| growing smoothly (`2.4e-06` at 32 channels to `3.1e-05` at 1024) and zero
NaN/Inf. Full output: [`m3-ldmatrix-channel-sweep-output-t4.txt`](m3-ldmatrix-channel-sweep-output-t4.txt);
analysis in `m3-block-results.md`'s "M3-10 follow-up" section.

**Conclusion: channel count alone is not the trigger.** The real block's residual-unit
convolutions run at exactly 256 channels — a case this sweep passed cleanly. Whatever triggers the
abort in the full block, it is not simply "wide enough conv2d."

### 2b. What remained unbisected when the decision to stop was made

At the time of the channel-count refutation, `m3-block-results.md` proposed an op-level bisection
ordered: (1) `ops.conv2d_transpose` at the real block's exact shape, in isolation; (2)
`ops.conv2d` at dilation=9 (the one untested dilation) at `seq_len=20`; (3) a reduced multi-op
composed graph; (4) the full CPU/GPU device-transfer boundary at the real shapes. **None of this
bisection was run.** Per the project owner's decision (recorded in §5 below), it is not going to
be — the root cause has since been established independently via upstream sources (§3), making a
local op-level bisection unnecessary to reach a conclusion, even though it would likely have
localized which specific op combination first hits the `ldmatrix` codegen path.

---

## 3. Root cause: established, upstream, and confirmed via multiple independent sources

This is not a hypothesis inferred from our own runs — it is confirmed against Modular's own forum,
public issue tracker, and an open pull request, all fetched and read directly.

### 3a. Modular community forum — the identical error string, same hardware

<https://forum.modular.com/t/having-issues-with-max-matmul-on-default-google-colab-gpu-t4/1658>

A separate user hit the exact same `ldmatrix.sync.aligned.m8n8.x4.b16` LLVM abort running MAX
Graph matmul on a default Google Colab T4 GPU. Elementwise ops (add/mul) and CPU execution worked
fine in that report; only the tensor-core matmul path failed. **Brad Larson, a Modular employee,
replied directly on the thread:**

> "we currently have Tensor Core support in the kernels for Ampere and newer NVIDIA GPUs, but
> Turing support was a new community addition and the Tensor Core operations haven't been
> modified to extend support back to that architecture."

The thread was closed **without a resolution and without a workaround offered** — confirming there
is no known runtime flag or configuration fix on that side either.

### 3b. `modular/modular` issue #4692 — open since 2025-05-25, still open

FP32 matmul on T4 hits the identical error string. Opened 2025-05-25; **still open as of this
document's writing (2026-08-25)** — over a year with no upstream fix landed.

### 3c. `modular/modular` issue #6653 — open since 2026-06-08, still open

The same error string, hit through `max serve` on T4 with three different real models
(gemma-3-1b-it, Qwen3-0.6B, Llama-3.2-3B) — i.e. this is not specific to a hand-built MAX graph; it
reproduces through Modular's own serving stack on the same hardware class. **A Modular team member
commented:**

> "We don't have many of the intrinsics configured for Turing GPUs"

This is the single most load-bearing quote for this document's conclusion: it is Modular's own
team characterizing the gap as a **backend configuration gap**, not a hardware limitation — the
`ldmatrix` intrinsic itself is architecturally valid on `sm_75` (Turing does have a matrix-load
path); MAX's Mojo/kernel backend simply has not been wired up to lower to it for that architecture
class.

### 3d. `modular/modular` PR #6659 — fixes #6653, open, NOT merged

Opened 2026-06-09, the day after #6653. This PR **programmatically emulates `ld_matrix` for
pre-Ampere architectures** in `mojo/stdlib/std/gpu/compute/mma.mojo`. Where the operation is a
genuine sm_80+ hardware requirement (bf16/tf32 MMA — actual tensor-core matrix-multiply-accumulate,
as opposed to the load intrinsic), it instead routes to naive SIMT compute paths
(`mha_gpu_naive`, and a naive matmul kernel in
`max/kernels/src/linalg/matmul/gpu/__init__.mojo`), gated by a compile-time
`compute<8.0` predicate. The PR states it was validated on real Tesla T4 hardware.

**As of this document's writing: not merged, and not in any released version.** This project's
pinned toolchain is **MAX 26.5.0** — the fix is not present in it, and there is no released MAX
version that contains it.

### 3e. `conv2d`/transposed-conv lowers through the same matmul pipeline

Modular's own documentation (`docs.modular.com/mojo/kernels/nn/conv_sm100/conv2d/im2col/`) and
their engineering blog post "Structured Mojo Kernels Part 3" describe `conv2d` as lowering through
an im2col transform into the same GEMM/matmul kernel pipeline that #4692/#6653/the forum thread all
hit. This is consistent with — though not a byte-for-byte proof of — why a full decoder block
(which chains `conv2d`-based residual-unit convolutions with a `conv2d_transpose`) can hit the same
`ldmatrix` wall that plain matmul and plain `conv2d`-serving hit, even though §2a's narrow
single-op sweep at moderate shapes did not.

### 3f. No runtime workaround exists

Checked explicitly, not assumed: there is no environment variable to force a naive/non-tensor-core
kernel path on NVIDIA hardware. The one documented precedent for this *shape* of override is
Apple-specific (`MODULAR_APPLE_M5_ALLOW_LOSSY_F32_MATMUL`), and no NVIDIA analogue exists in the
pinned MAX 26.5.0. PR #6659's routing decision is a **compile-time** predicate
(`compute<8.0`), not a runtime flag — so even on a from-source Mojo/MAX checkout of that
unmerged PR, there would be no toggle; the architecture-conditional code path is baked in at
compile time based on target GPU compute capability.

---

## 4. What is NOT confirmed (explicitly — do not read the above as covering these)

- **No documented tile-size threshold for tensor-core vs. naive kernel selection.** Whether MAX's
  kernel-selection heuristic has any size-based cutoff (as opposed to being purely
  architecture-gated) is not established by anything read for this document. §2a's channel sweep
  found no such cutoff up to 1024 channels for `ops.conv2d` alone, but that does not generalize to
  every op/shape combination, and no upstream source confirms or denies a tile-size threshold
  either way.
- **No confirmation of whether FP32 inputs are silently routed through a TF32 tensor-core path.**
  The forum thread and #4692 both involve FP32 workloads hitting the tensor-core `ldmatrix` path,
  which suggests FP32 can reach that path on this hardware, but no source read for this document
  explicitly states the routing/promotion mechanism (e.g. whether MAX internally treats FP32 matmul
  as eligible for a TF32-precision tensor-core kernel by default). This is left as unconfirmed, not
  asserted either way.
- **No op-level bisection was run** identifying which specific op or op-combination in the full
  block first triggers the abort (conv2d_transpose alone vs. dilation=9 conv2d vs. graph fusion vs.
  the CPU/GPU transfer boundary) — see §2b. Given the confirmed upstream/architectural root cause,
  this bisection would be informative about *which exact op path* hits the gap in this specific
  graph, but would not change the conclusion that the underlying cause is upstream and unfixable
  from this project's side; it was not run because that additional detail is no longer decision-
  relevant given the stop decision in §5.

---

## 5. Decision and status

**Decision (project owner, 2026-08-25): stop further T4 verification for M3.** The op-level
bisection outlined in §2b is cancelled. This is not a workaround-and-move-on decision — no
workaround exists on this side to apply (§3f) — it is a decision to stop spending further effort
confirming the shape of a bug that is already independently confirmed, open, and un-mergeable from
this project's side, and to redirect effort to what MAX/Mojo demonstrably does support (M1/Metal,
per every M3-1 through M3-9 result in `m3-block-results.md`).

**M3-10 is BLOCKED, not done, not skipped.** Its own done-criteria ("the Tesla T4 run's raw output
is committed... and every M1 result above is confirmed or contradicted per-case") are genuinely not
met: only M3-5's synthetic-weight case was exercised on T4 (and it aborts), M3-6/M3-7/M3-9 were
never attempted on T4 given the M3-5 abort, and M3-1's mixed-device-placement question was answered
only implicitly (device placement is reported in the stdout above and is not itself the cause of
the abort — the graph compiles device placement successfully before the LLVM backend fails on
codegen). Per `AGENTS.md`'s "an honest partial result is valid... never turn a load-only success
into an inference pass" and "do not close an issue while the original user-requested outcome or any
acceptance criterion remains incomplete," this is recorded as **blocked by an external, upstream
cause**, not as completed work, and M3-10's checkbox in `m3-plan.md` is marked accordingly rather
than checked off.

---

## 6. Realistic options for the future (unverified hypotheses, not decisions)

None of the following have been tried or validated. They are recorded as candidate directions for
whoever picks this up again, explicitly flagged as **unproven**:

- **Force the problematic operation(s) onto CPU.** `conv2d_transpose` is already CPU-placed for an
  unrelated reason (`CUDNN_STATUS_ALLOC_FAILED` on T4's GPU path, per `m2-convtranspose1d-
  results.md`); it is not established whether CPU-placing the `conv2d`-based residual-unit
  convolutions too would avoid the `ldmatrix` path, nor what the performance cost of an
  all-CPU-conv graph on T4 would be. Untested.
- **Test on Ampere-or-newer NVIDIA hardware instead (A100, L4).** Per Brad Larson's forum reply
  (§3a), tensor-core kernels are supported on Ampere+; an A100 or L4 Colab/cloud instance would not
  be expected to hit this specific gap. This has not been attempted — no such run has been made in
  this project, and it would be a new hardware target, not a fix for T4.
- **Wait for PR #6659 (or an equivalent fix) to merge and ship in a released MAX version.** The PR
  exists, targets exactly this failure, and reports validation on real T4 hardware — but it is
  unmerged, and there is no committed timeline from Modular for when or whether it lands, nor
  whether the merged form will behave identically to the current PR diff.

---

## 7. Open item: further verification on non-Apple hardware (intent only, not planned in detail)

Recorded so it is not lost, per the project owner's direction: once the rest of the stack is
otherwise ready, verification on hardware other than Apple Silicon is intended to be attempted on
**Kaggle Notebooks** (already referenced in `README.md`'s Kaggle section for the existing
vLLM-Omni benchmark), rather than Colab.

- Kaggle's available GPUs include (at least) **P100** and **T4×2**. This is **not** a
  from-first-principles fix for the finding above: Kaggle's T4 is the same Turing/`sm_75`
  architecture as Colab's T4 and is expected to hit the identical `ldmatrix` abort for the same
  upstream reason — using Kaggle does not itself avoid this issue if a T4 session is what gets
  assigned.
- P100 is Pascal (`sm_60`), which has no tensor cores at all. The **unverified hypothesis** is that
  a tensor-core-free architecture would not enter the `ldmatrix`/tensor-core codegen path this
  document's root cause is about, and so might not hit this specific abort — but this has not been
  tested anywhere in this project, and Pascal predates even the Ampere+ tensor-core requirement
  discussed in §3, so it is a different (and equally unverified) question whether MAX 26.5.0's
  Pascal support has its own gaps.
- Whichever GPU a future Kaggle session actually provides, the first required check before relying
  on it is its **compute capability**: per §3a/§3d, tensor-core kernel support in this MAX version
  is scoped to Ampere and newer, i.e. **compute capability >= 8.0**. A GPU below that threshold
  (Turing `sm_75`, Pascal `sm_60`, or anything else pre-Ampere) is not established as safe by
  anything in this document — it would need to be checked against the same failure mode fresh, not
  assumed clear because it isn't a T4.
- This is an intent, not a plan: no notebook, script, or task has been created for this yet, and no
  timeline is set. The GPU-detection-first discipline `README.md`'s Kaggle section already
  requires (print GPU name/VRAM/compute capability before trusting any hardware-specific code path)
  applies here too, and should gate whether any of M3-5 through M3-9 are even attempted on whatever
  GPU a Kaggle session actually assigns.

---

## 8. References

- Modular community forum: <https://forum.modular.com/t/having-issues-with-max-matmul-on-default-google-colab-gpu-t4/1658>
  (Brad Larson, Modular: Tensor Core support exists for Ampere+, Turing support "was a new
  community addition and the Tensor Core operations haven't been modified to extend support back
  to that architecture." Thread closed without a resolution or workaround.)
- `modular/modular` issue #4692 (opened 2025-05-25, **open**): FP32 matmul on T4, identical
  `ldmatrix` error string.
- `modular/modular` issue #6653 (opened 2026-06-08, **open**): identical error via `max serve` on
  T4 with gemma-3-1b-it / Qwen3-0.6B / Llama-3.2-3B. Modular team comment: "We don't have many of
  the intrinsics configured for Turing GPUs."
- `modular/modular` PR #6659 (opened 2026-06-09, **not merged**): fixes #6653 by emulating
  `ld_matrix` for pre-Ampere architectures and routing genuine sm_80+-only bf16/tf32 MMA ops to
  naive SIMT kernels (`mha_gpu_naive`, a naive matmul kernel in
  `max/kernels/src/linalg/matmul/gpu/__init__.mojo`) via a compile-time `compute<8.0` predicate.
  Validated on real T4 hardware by the PR author. Not present in MAX 26.5.0 (this project's pinned
  version) or any released version as of this writing.
- `docs.modular.com/mojo/kernels/nn/conv_sm100/conv2d/im2col/` and Modular's engineering blog post
  "Structured Mojo Kernels Part 3": describe `conv2d`'s lowering through im2col into the same
  matmul/GEMM kernel pipeline implicated above.
- This project's own prior findings: `m3-block-results.md` M3-5 through M3-9 sections (all M1/Metal
  PASSED results this T4 failure does not contradict — the abort is Turing/T4-specific, not a
  regression from any M1 finding), and the "M3-10 follow-up" section (channel-count hypothesis
  refutation, `m3_ldmatrix_channel_sweep.py`, `m3-ldmatrix-channel-sweep-output-t4.txt`).
- Raw T4 abort output for all 6 seeds: [`m3-block-output-t4.txt`](m3-block-output-t4.txt) (this
  task).
