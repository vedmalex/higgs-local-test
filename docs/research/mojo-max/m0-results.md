# M0 — Mojo/MAX hardware probe on this project's M1 (issue #57)

Date: 2026-08-24. Branch: `research/57-m0-hardware-probe`, follow-up on
`research/57-m0-gpu-unblocked` after the host OS was upgraded.

## Verdict (updated after macOS upgrade to 26.6.2)

| Item | Status |
| --- | --- |
| Install stable Mojo 1.0.0 / MAX 26.5 on this M1 | **PASSED** |
| Minimal accelerator program executed **on the GPU** | **PASSED** (was BLOCKED on macOS 14.6.1) |
| Run an existing supported MAX model/pipeline on GPU | **PASSED** — MAX's own shipped kernels (matmul incl. BF16) execute on the Apple GPU |
| Numerical suite (matmul/softmax/RMSNorm/SiLU/NaN-Inf/FP16-vs-FP32) | **PASSED on CPU and GPU** |

**M0 is now complete on this machine's Apple-GPU side.** The original blocker
(macOS 14.6.1 < required Sequoia 15) was resolved by upgrading the host to
macOS 26.6.2. A second, unrelated blocker then appeared and was also resolved
— see "Second blocker" below — before the probe could actually execute on
the GPU. M1 (the vLLM-Omni <-> MAX responsibility map) can now start using
this M1's GPU results as real evidence. The T4/Colab side of M0 is still
outstanding (untouched by either Apple-specific blocker).

## Second blocker found and resolved: Xcode/Metal toolchain incompatibility

Upgrading the OS past Sequoia 15 did not immediately unblock the GPU: the
still-installed Xcode 16.2 was incompatible with macOS 26.6.2 in a way that
broke the entire Metal toolchain (`xcodebuild`/`xcrun`/`metal` all crashed
with a `CoreDevice` symbol-lookup error, `_XPCTypeBool` not found). Mojo's
`mojo run` failed at metallib compilation with `Metal Compiler failed to
compile metallib`.

Diagnosis and fix, in order:

1. `xcode-select -s /Library/Developer/CommandLineTools` (switching to the
   already-present standalone Command Line Tools 26.6 package) did **not**
   help — the standalone CLT package does not ship the `metal` compiler
   binary at all, only headers/specs. `metal` only exists inside a full
   Xcode.app install.
2. Xcode.app itself had to be updated from 16.2 to 26.6 (manual step: App
   Store, requires interactive sign-in — not automatable from the agent).
   After updating, `xcodebuild -version` worked, but launching `Xcode.app`
   itself crashed instantly with `SIGABRT` in `main.cold.1`, before loading
   any of its own frameworks — consistent with an unaccepted license /
   un-run first-launch setup blocking a background launch.
3. Fix: `sudo xcodebuild -license accept` then `sudo xcodebuild
   -runFirstLaunch` (both require an interactive terminal for the sudo
   password — not automatable from the agent's sandboxed shell either).
4. Even after that, `metal` failed with `cannot execute tool 'metal' due to
   missing Metal Toolchain; use: xcodebuild -downloadComponent
   MetalToolchain` — modern Xcode (17+ line, shipped as "26.6" here) splits
   the Metal compiler out into a separately downloaded component. Running
   `xcodebuild -downloadComponent MetalToolchain` fixed it: `xcrun -sdk
   macosx metal --version` then reported `Apple metal version 32023.883`.

None of steps 2–4 could be performed by the agent alone: steps requiring
`sudo` need an interactive TTY for the password, and the Xcode.app update
itself needs an authenticated App Store / Apple Developer download. The user
performed these manually; the agent only diagnosed each failure and directed
the next command.

**Toolchain now in place:** macOS 26.6.2 (25G83), Xcode 26.6 (24959, build
17F113), Metal compiler 32023.883 (target `air64-apple-darwin25.6.0`),
Mojo 1.0.0 (ed45d567), MAX 26.5.0 — identical Mojo/MAX versions to the first
(CPU-only) run, so the two runs are directly comparable; only the OS/Xcode/
Metal layer changed.

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

## GPU: detected, enumerated, and now executing correctly

Device detection and enumeration, unchanged from the first run:

```text
has_accelerator            = True
has_apple_gpu_accelerator  = True
has_nvidia_gpu_accelerator = False
has_amd_gpu_accelerator    = False
device name = Apple M1
device api  = metal
```

With the toolchain fixed (see above), every GPU step that previously failed at
Metal-function creation now runs and produces correct numbers, matching the
CPU reference:

```text
gpu vadd c[0] = 3.0 (expected 3.0)
  [PASSED] GPU kernel actually executed on device
gpu matmul fp32: nan/inf 0 max|err| vs CPU fp32 0.0
  [PASSED] GPU matmul FP32 matches CPU reference
gpu matmul fp16 (acc fp32): nan/inf 0 max|err| vs CPU fp32 0.0038223267
  [PASSED] GPU matmul FP16 with FP32 accumulation finite
gpu matmul bf16 (acc fp32): nan/inf 0
  [PASSED] GPU matmul BF16 supported
```

Notable: **BF16 matmul runs and is finite on the Apple M1 GPU.** This matters
for the #48/#52 dtype-policy question specifically because T4 (`sm_75`,
Turing) has **no BF16 hardware support at all** — so a MAX/Mojo precision
policy for Higgs/Qwen cannot assume BF16 portability between Apple Silicon and
T4 even if both run through MAX; the policy still needs to select FP16 (with
FP32 accumulation where numerically required, per the CPU findings below) on
T4 specifically.

The FP16 GPU matmul error (`0.0038223267`) matches the CPU FP16-with-FP32-
accumulation result from the first run exactly, confirming the GPU path uses
the same numerics as the validated CPU path — the earlier CPU-only findings
about FP32-accumulation-insufficiency and FP16 RMSNorm silent-zero failure
(below) can now be treated as GPU-relevant, not just theoretical.

Conclusion: the toolchain, the install, the GPU hardware and the probe code
are all fine, and now so is the Xcode/Metal layer. Both original-plan
blockers (macOS version, then the Xcode/Metal-toolchain incompatibility
surfaced by the OS jump) are resolved on this M1.

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

1. **Apple M1 side: done.** GPU probe passes end-to-end on macOS 26.6.2 /
   Xcode 26.6 / Metal toolchain 32023.883 / Mojo 1.0.0 / MAX 26.5.0.
2. **T4 / Colab (`sm_75`)**: still outstanding — run the same file there. That
   side of M0 was never touched by either Apple-specific blocker and is where
   the #48-relevant, no-BF16-hardware GPU numbers will come from. Watch for
   the driver-580+ requirement and the `MODULAR_NVPTX_COMPILER_PATH` escape
   hatch noted in the issue.
3. M1 (the vLLM-Omni <-> MAX responsibility map, per the issue's revised
   framing) may now start using the Apple-GPU results as real on-device
   evidence, in parallel with getting the T4 run. Do not treat the T4 side as
   started until it actually executes on that device.
4. Disk: after this run, `.mojo-probe-stable` still costs ~1.9 GiB and remains
   gitignored/regenerable. Free space is no longer the acute constraint it was
   before the cache cleanup (issue-unrelated), but MAX model serving will need
   "significantly more memory" per the requirements page once real
   checkpoints are involved.
