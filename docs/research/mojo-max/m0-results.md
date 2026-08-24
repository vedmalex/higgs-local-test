# M0 — Mojo/MAX hardware probe on this project's M1 (issue #57)

Date: 2026-08-24. Branch: `research/57-m0-hardware-probe`.

## Verdict

| Item | Status |
| --- | --- |
| Install stable Mojo 1.0.0 / MAX 26.5 on this M1 | **PASSED** |
| Minimal accelerator program executed **on the GPU** | **BLOCKED** — macOS 14.6.1 < required macOS 15 |
| Run an existing supported MAX model/pipeline on GPU | **BLOCKED** — same cause |
| Numerical suite (matmul/softmax/RMSNorm/SiLU/NaN-Inf/FP16-vs-FP32) | **PASSED on CPU**, **BLOCKED on GPU** |

**M0 is not complete on this machine, and M1–M6 must not start from it.** The
single blocker is the host OS version, not the hardware, the toolchain or the
code. The GPU is a supported M1; the probe program compiles cleanly and its
numerics are validated. Only on-device execution is unavailable.

## Hardware and toolchain actually measured

| Property | Value | Source |
| --- | --- | --- |
| Chip | Apple M1 | `sysctl -n machdep.cpu.brand_string` |
| Kernel | `RELEASE_ARM64_T8103`, Darwin 23.6.0 | `uname -a` |
| Unified memory | 16 GiB (17179869184 bytes) | `sysctl -n hw.memsize` |
| **macOS** | **14.6.1 (23G93) — Sonoma** | `sw_vers` |
| Xcode | 16.2 (16C5032a) | `xcodebuild -version` |
| macOS SDK | 15.2 | `xcrun --show-sdk-version` |
| Metal compiler | Apple metal 32023.404, target `air64-apple-darwin23.6.0` | `xcrun -sdk macosx metal --version` |
| Free disk at start | 17 GiB of 926 GiB (99% full) | `df -h` |

Requirement check against <https://mojolang.org/docs/requirements> (note: the
`docs.modular.com/mojo/*` URLs now 307-redirect to `mojolang.org/docs/*`):

- required: *"macOS Sequoia (15) or later"* — **this host has 14.6.1. FAIL.**
- required: *"Xcode or Xcode Command Line Tools 16 or later on macOS"* — 16.2. PASS.
- required GPU: M1–M5 "Known compatible" — M1. PASS.
- required memory: 8 GiB minimum for Mojo development — 16 GiB. PASS.

So the documented blocker cited in issue #57 is real and current, and it is the
**macOS version alone**. Xcode 16+ is already satisfied here.

## What was actually installed

The current official install path is `pixi` (or `uv`); the old `modular install`
/ `magic` flow is gone. Neither `pixi` nor `magic` was present on this machine.

```bash
# native arm64 — the login shell here runs under Rosetta (sysctl.proc_translated=1),
# so every command is wrapped in `arch -arm64`
arch -arm64 /bin/bash -c 'curl -fsSL https://pixi.sh/install.sh | bash'   # pixi 0.77.0

mkdir -p .mojo-probe-stable && cd .mojo-probe-stable
arch -arm64 pixi init . -c https://conda.modular.com/max/ -c conda-forge
arch -arm64 pixi add modular          # resolved modular >=26.5.0,<27
```

Verified versions:

```text
$ arch -arm64 pixi run mojo --version
Mojo 1.0.0 (ed45d567)
$ arch -arm64 pixi run max --version
MAX 26.5.0
```

**The install succeeded despite the unmet macOS requirement.** That is worth
recording: the package manager does not gate on OS version, so "it installed"
is not evidence of support. A nightly channel
(`https://conda.modular.com/max-nightly/`, Mojo 1.1.0.dev2026082305 / MAX
26.6.0.dev2026082305) was also installed and tested; it fails identically, so it
was deleted afterwards to reclaim disk. Each environment costs ~1.9 GB, which
matters on a host that started at 99% full.

## GPU: detected, enumerated, and unable to execute

Device detection and enumeration work fully:

```text
has_accelerator            = True
has_apple_gpu_accelerator  = True
has_nvidia_gpu_accelerator = False
has_amd_gpu_accelerator    = False
device name = Apple M1
device api  = metal
```

The MAX Python driver agrees: `driver.accelerator_count() == 1`,
`driver.Accelerator()` → `Device(type=gpu,id=0)`, `driver.accelerator_api()` →
`metal`.

Every attempt to *run* anything on that device fails at the same point — Metal
function creation:

```text
[BLOCKED] GPU vector-add failed: At max/mojo/max/gpu/host/_device_context_extras.mojo:168:17:
  Failed to create Metal function: m0_smoke_test_gpu_vadd_TileTe6A6AoA6A6A_cbbeb29f3ee4a585
[BLOCKED] GPU matmul fp32 failed: ... Failed to create Metal function: m0_smoke_test_gpu_matmul_...
[BLOCKED] GPU matmul fp16 failed: ... Failed to create Metal function: m0_smoke_test_gpu_matmul_...
[SKIPPED/BLOCKED] GPU matmul bf16 unavailable: ... Failed to create Metal function: ...
```

This is **not** specific to hand-written kernels. MAX's own shipped kernels fail
the same way when a graph is placed on the accelerator:

```text
matmul  -> kernel "matmul_mogg":            Failed to create Metal function: gemm_kernel_apple_8
softmax -> kernel "reduce_softmax_mogg":    Failed to create Metal function: softmax_tem
silu*b  -> kernel "Elementwise_silu_mul_mogg": Failed to create Metal function: Eleme...
```

Note the kernel name `gemm_kernel_apple_8` — MAX does ship an Apple-specific
GEMM path; it simply cannot be loaded by this OS's Metal runtime.

### The install is sound — controls prove the failure is Metal-specific

- Graph *construction and compilation* for a GPU `DeviceRef` succeeds; only
  execution fails. So this is a runtime library-load failure, not a compiler or
  codegen failure.
- The Metal toolchain itself is healthy: `xcrun metal --version` works and
  `metallib` is present. AIR compilation is not the problem.
- The same MAX graph on the **CPU** device executes correctly:
  `CPU MAX graph matmul fp32: OK max|err| vs fp32 = 0 nan/inf=0`.
- The Mojo CPU numerical suite runs to completion with 16/16 checks passing.

Conclusion: the toolchain, the install, the GPU hardware and the probe code are
all fine. macOS 14.6.1's Metal runtime cannot load the Metal library that Mojo
1.0 / MAX 26.5 produce for Apple GPUs. This matches the documented macOS 15+
requirement exactly, and reproduces on both stable 26.5 and nightly 26.6.

**To unblock: upgrade this host to macOS 15 (Sequoia) or later.** Nothing else is
missing. No workaround was attempted, and none should be.

## Numerical results (CPU reference, real numbers)

Run: `pixi run mojo run docs/research/mojo-max/m0_smoke_test.mojo`.
Full captured output: [`m0-output-m1.txt`](m0-output-m1.txt).
Inputs come from a fixed LCG, so T4 and M1 runs are bit-comparable.

These numbers are the **CPU** path. They validate the probe and answer the dtype
questions in #57, but they are **not** evidence about GPU behaviour on either
target.

| Operation | Result |
| --- | --- |
| matmul fp32 (acc fp32) | min -8.778667, max 9.596451, nan/inf 0 |
| matmul fp16 (acc fp16) | nan/inf 0, max abs err 0.021327019, max rel err 0.061863705 |
| matmul fp16 (acc **fp32**) | nan/inf 0, max abs err **0.0038223267**, max rel err 0.014604608 |
| matmul bf16 (acc fp32) | nan/inf 0, max abs err 0.033950806, max rel err 0.1079008 |
| softmax fp32 | sum 1.0000005, max 0.11529518, nan/inf 0 |
| softmax fp16 (acc fp16) | nan/inf 0, max abs err 0.0003106445 |
| softmax fp16 (acc fp32) | nan/inf 0, max abs err 0.00015059113 |
| softmax, extreme scores (-60..+130) | fp32 nan/inf 0, fp16 nan/inf 0, max abs err 1.4364719e-05 |
| RMSNorm fp32 | min -1.7403387, max 1.7506354, nan/inf 0 |
| RMSNorm fp16 (acc fp16) | nan/inf 0, max abs err 0.0029147863 |
| RMSNorm fp16 (acc fp32) | nan/inf 0, max abs err 0.00080513954 |
| SiLU gated MLP fp32 | min -30.964148, max 33.96759, nan/inf 0 |
| SiLU gated MLP fp16 | nan/inf 0, max abs err 0.0374012, max rel err 0.0017562123 |

Per-check status, as emitted by the program: **16 PASSED, 0 FAILED**, covering
matmul FP32/FP16/BF16 finiteness, FP32-accumulation improvement, the overflow
regime, softmax, RMSNorm, SiLU and NaN/Inf detection.

### Two findings that bear directly on the Higgs T4 failure (#48)

**1. FP32 accumulation is necessary but not sufficient.** In a deliberately
large-magnitude matmul whose true FP32 result reaches 77731.24 — beyond the FP16
finite range of 65504 — FP32 accumulation reduced but did not eliminate
overflow:

```text
large-magnitude matmul: fp32 true max 77731.24 (FP16 finite range is 65504)
  fp16 acc fp16: nan/inf 11 zeros 0
  fp16 acc fp32: nan/inf  6 zeros 0
```

Accumulating in FP32 cannot rescue a value whose *true* result does not fit the
FP16 **storage** dtype — the final downcast still overflows. So for Code2Wav it
is not enough to promote accumulation; wherever activations genuinely exceed the
FP16 range, the storage dtype must be raised too, or the tensor rescaled.

**2. FP16 RMSNorm fails *silently*, and a NaN/Inf scan will not catch it.** With
a large input (sum of squares ~5e6, past the FP16 range) the FP16 sum of squares
overflows to +Inf, so `1/sqrt(Inf)` collapses to 0 and **every** output becomes
exactly zero — finite, and completely wrong:

```text
rmsnorm large input (sum-of-squares ~5e6, beyond FP16 range):
  acc fp16: nan/inf 0 zeros 256 max|err| vs fp32 1.7506355
  acc fp32: nan/inf 0 zeros   0 max|err| vs fp32 0.0008292198
```

This is the same class of silent corruption as the `-32768` waveform in #48: a
saturated/zeroed tensor rather than a NaN. Instrumentation for the M2/M3 work
must compare against an FP32 reference and watch for all-zero and saturated
tensors — counting NaN/Inf alone would report this failure as healthy. Here FP32
accumulation repaired it completely (max abs err 8.3e-4).

Reassuringly, max-subtracted softmax was robust in FP16 even with scores from
-60 to +130, so the softmax is unlikely to be the origin of the T4 problem.

## Mojo 1.0 syntax notes (post-1.0; verified by compiling)

The official Modular skills were installed and used — `mojo-syntax` and
`mojo-gpu-fundamentals` in particular. Pretrained/older syntax is wrong in ways
that matter, and even the skills lag stable 26.5 in one place:

- `fn` is **removed** — it is a hard parse error. Every function is `def`.
- `alias` → `comptime`; `@parameter if/for` → `comptime if/for`;
  `constrained(...)` → `comptime assert ...`.
- `ref` and `var` are hard keywords and cannot be used as identifiers at all.
- `List` is not `ImplicitlyCopyable`: `return out` must be `return out^`.
- Generic functions calling `exp`/`sqrt` on `Scalar[acc]` need an explicit
  constraint clause: `... -> List[Scalar[dt]] where acc.is_floating_point():`.
- **Correction to `mojo-gpu-fundamentals`:** it says plain `Int` scalar kernel
  args "are fine". On stable 26.5 that fails with
  `Int and UInt do not conform to DevicePassable; use a fixed-width type such as
  Int32 or Int64 instead`. Use `Int32`/`Int64` for kernel scalar arguments.
- `max.driver` has no `Tensor` in 26.5; host→device transfer is
  `max.driver.Buffer.from_numpy(a).to(dev)`, and graph inputs need a
  `DeviceRef` so tensors land on the intended device.

## Reproducing

```bash
export PATH="$HOME/.pixi/bin:$PATH"
cd .mojo-probe-stable
arch -arm64 pixi run mojo run ../docs/research/mojo-max/m0_smoke_test.mojo
```

The environment is git-ignored (~1.9 GB). Recreate it with the two `pixi`
commands above. The probe is written so the GPU and CPU sections fail
independently: each GPU step is individually guarded, so on a T4 a later failure
still leaves the earlier evidence intact, and a host with no usable GPU still
produces the full numerical report.

## What M0 still needs

1. **This M1: upgrade to macOS 15+**, then re-run the probe unchanged. Until
   then Apple-GPU numbers for #57 cannot be produced here, and MLX-Audio remains
   the only working local path (consistent with `AGENTS.md`).
2. **T4 / Colab (`sm_75`)**: run the same file. That side of M0 is untouched by
   this blocker and is where the #48-relevant GPU numbers will come from. Watch
   for the driver-580+ requirement and the `MODULAR_NVPTX_COMPILER_PATH` escape
   hatch noted in the issue.
3. Only when at least one target executes on-device should M1 begin. Do not
   treat this CPU-only run as M1 evidence.
4. Disk on this host is the next practical constraint: 13 GiB free after one
   environment. MAX model serving needs "significantly more memory" per the
   requirements page, and a GenAI checkpoint would not fit alongside.
