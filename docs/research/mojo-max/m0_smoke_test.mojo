# M0 hardware/numerical probe for issue #57 (Mojo/MAX feasibility spike).
#
# Purpose: run the SAME program on Apple M1 and on NVIDIA T4 (sm_75) and record
# device execution, dtype behaviour and the numerical properties that the Higgs
# Code2Wav FP16 failure (#48) depends on: matmul, softmax, RMSNorm, SiLU gated
# MLP, attention-like score ranges, NaN/Inf behaviour and explicit FP32
# accumulation/upcast.
#
# Written against stable Mojo 1.0.0 / MAX 26.5.0 syntax.
# Run with:  pixi run mojo run docs/research/mojo-max/m0_smoke_test.mojo
#
# The CPU section is the numerical reference and always runs. The GPU section is
# attempted separately and reports its own failure honestly instead of aborting,
# so a machine that cannot execute GPU kernels still yields a usable report.

from std.math import exp, sqrt, isnan, isinf, ceildiv
from std.sys import has_accelerator
from std.sys.info import (
    has_apple_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    has_amd_gpu_accelerator,
)
from std.gpu import global_idx, thread_idx, block_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

# ---------------------------------------------------------------- dimensions

comptime M = 64
comptime N = 64
comptime K = 64
comptime SEQ = 128
comptime HID = 256
comptime TILE = 16

comptime g_layout = row_major[M, N]()
comptime v_layout = row_major[1024]()

# ------------------------------------------------------- deterministic input
# An LCG keeps the inputs bit-identical across machines, so M1 and T4 numbers
# are directly comparable.


struct Rng(Copyable, Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next_unit(mut self) -> Float32:
        # returns a value in [-1, 1)
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        var bits = (self.state >> 40).cast[DType.float32]()
        return bits / Float32(8388608.0) - Float32(1.0)


def fill[dt: DType](count: Int, scale: Float32, seed: UInt64) -> List[Scalar[dt]]:
    var out = List[Scalar[dt]](capacity=count)
    var rng = Rng(seed)
    for _ in range(count):
        out.append((rng.next_unit() * scale).cast[dt]())
    return out^


# -------------------------------------------------------------- diagnostics


def count_bad[dt: DType](v: List[Scalar[dt]]) -> Int:
    var bad = 0
    for i in range(len(v)):
        var x = v[i].cast[DType.float32]()
        if isnan(x) or isinf(x):
            bad += 1
    return bad


def min_of[dt: DType](v: List[Scalar[dt]]) -> Float32:
    var m = Float32.MAX_FINITE
    for i in range(len(v)):
        var x = v[i].cast[DType.float32]()
        if not isnan(x) and x < m:
            m = x
    return m


def max_of[dt: DType](v: List[Scalar[dt]]) -> Float32:
    var m = -Float32.MAX_FINITE
    for i in range(len(v)):
        var x = v[i].cast[DType.float32]()
        if not isnan(x) and x > m:
            m = x
    return m


def max_abs_diff[dt: DType](
    base: List[Scalar[DType.float32]], got: List[Scalar[dt]]
) -> Float32:
    var worst = Float32(0.0)
    for i in range(len(base)):
        var a = base[i]
        var b = got[i].cast[DType.float32]()
        if isnan(b) or isinf(b):
            return Float32.MAX_FINITE
        var d = abs(a - b)
        if d > worst:
            worst = d
    return worst


def max_rel_diff[dt: DType](
    base: List[Scalar[DType.float32]], got: List[Scalar[dt]], floor: Float32
) -> Float32:
    var worst = Float32(0.0)
    for i in range(len(base)):
        var a = base[i]
        var b = got[i].cast[DType.float32]()
        if isnan(b) or isinf(b):
            return Float32.MAX_FINITE
        var denom = abs(a)
        if denom < floor:
            continue
        var d = abs(a - b) / denom
        if d > worst:
            worst = d
    return worst


def count_zeros[dt: DType](v: List[Scalar[dt]]) -> Int:
    var z = 0
    for i in range(len(v)):
        if v[i].cast[DType.float32]() == Float32(0.0):
            z += 1
    return z


def report(name: String, ok: Bool):
    if ok:
        print("  [PASSED]", name)
    else:
        print("  [FAILED]", name)


# ------------------------------------------------------------- CPU reference
# `dt` is the storage/compute dtype, `acc` the accumulation dtype. Keeping them
# independent is exactly the storage/compute/accumulation separation that #57
# asks us to verify.


def matmul[
    dt: DType, acc: DType
](A: List[Scalar[dt]], B: List[Scalar[dt]]) -> List[Scalar[dt]]:
    var C = List[Scalar[dt]](capacity=M * N)
    for _ in range(M * N):
        C.append(Scalar[dt](0))
    for i in range(M):
        for j in range(N):
            var s = Scalar[acc](0)
            for k in range(K):
                s += A[i * K + k].cast[acc]() * B[k * N + j].cast[acc]()
            C[i * N + j] = s.cast[dt]()
    return C^


def softmax[
    dt: DType, acc: DType
](row: List[Scalar[dt]]) -> List[Scalar[dt]] where acc.is_floating_point():
    var n = len(row)
    var out = List[Scalar[dt]](capacity=n)
    # max subtraction, the standard stabilisation
    var mx = -Float32.MAX_FINITE.cast[acc]()
    for i in range(n):
        var x = row[i].cast[acc]()
        if x > mx:
            mx = x
    var total = Scalar[acc](0)
    var tmp = List[Scalar[acc]](capacity=n)
    for i in range(n):
        var e = exp(row[i].cast[acc]() - mx)
        tmp.append(e)
        total += e
    for i in range(n):
        out.append((tmp[i] / total).cast[dt]())
    return out^


def rmsnorm[
    dt: DType, acc: DType
](x: List[Scalar[dt]], eps: Float32) -> List[Scalar[dt]] where acc.is_floating_point():
    var n = len(x)
    var ss = Scalar[acc](0)
    for i in range(n):
        var v = x[i].cast[acc]()
        ss += v * v
    var inv = Scalar[acc](1) / sqrt(ss / Scalar[acc](n) + eps.cast[acc]())
    var out = List[Scalar[dt]](capacity=n)
    for i in range(n):
        out.append((x[i].cast[acc]() * inv).cast[dt]())
    return out^


def silu_gated[
    dt: DType, acc: DType
](gate: List[Scalar[dt]], up: List[Scalar[dt]]) -> List[Scalar[dt]] where acc.is_floating_point():
    var n = len(gate)
    var out = List[Scalar[dt]](capacity=n)
    for i in range(n):
        var g = gate[i].cast[acc]()
        var silu = g / (Scalar[acc](1) + exp(-g))
        out.append((silu * up[i].cast[acc]()).cast[dt]())
    return out^


# ------------------------------------------------------------- GPU kernels
# These compile for whichever accelerator is present. `acc` is the explicit
# accumulation dtype so the FP32-accumulation question can be tested on device.


def gpu_vadd(
    a: TileTensor[DType.float32, type_of(v_layout), MutAnyOrigin],
    b: TileTensor[DType.float32, type_of(v_layout), MutAnyOrigin],
    c: TileTensor[DType.float32, type_of(v_layout), MutAnyOrigin],
    size: Int32,
):
    var tid = global_idx.x
    if tid < Int(size):
        c[tid] = a[tid] + b[tid]


def gpu_matmul[
    dt: DType, acc: DType
](
    A: TileTensor[dt, type_of(g_layout), MutAnyOrigin],
    B: TileTensor[dt, type_of(g_layout), MutAnyOrigin],
    C: TileTensor[dt, type_of(g_layout), MutAnyOrigin],
):
    comptime assert A.flat_rank == 2, "A must be rank 2"
    comptime assert B.flat_rank == 2, "B must be rank 2"
    comptime assert C.flat_rank == 2, "C must be rank 2"
    var row = block_idx.y * TILE + thread_idx.y
    var col = block_idx.x * TILE + thread_idx.x
    if row < M and col < N:
        var s = Scalar[acc](0)
        for k in range(K):
            var av = rebind[Scalar[dt]](A[row, k]).cast[acc]()
            var bv = rebind[Scalar[dt]](B[k, col]).cast[acc]()
            s += av * bv
        C[row, col] = rebind[C.ElementType](s.cast[dt]())


def run_gpu_vadd(ctx: DeviceContext) raises -> Float32:
    var a_buf = ctx.enqueue_create_buffer[DType.float32](1024)
    var b_buf = ctx.enqueue_create_buffer[DType.float32](1024)
    var c_buf = ctx.enqueue_create_buffer[DType.float32](1024)
    a_buf.enqueue_fill(1.0)
    b_buf.enqueue_fill(2.0)
    var a = TileTensor(a_buf, v_layout)
    var b = TileTensor(b_buf, v_layout)
    var c = TileTensor(c_buf, v_layout)
    ctx.enqueue_function[gpu_vadd](
        a, b, c, Int32(1024), grid_dim=ceildiv(1024, 256), block_dim=256
    )
    ctx.synchronize()
    var got: Float32
    with c_buf.map_to_host() as host:
        var t = TileTensor(host, v_layout)
        got = rebind[Scalar[DType.float32]](t[0])
    return got


def run_gpu_matmul[
    dt: DType, acc: DType
](ctx: DeviceContext, scale: Float32) raises -> List[Scalar[DType.float32]]:
    var host_a = fill[dt](M * K, scale, 11)
    var host_b = fill[dt](K * N, scale, 22)
    var a_buf = ctx.enqueue_create_buffer[dt](M * K)
    var b_buf = ctx.enqueue_create_buffer[dt](K * N)
    var c_buf = ctx.enqueue_create_buffer[dt](M * N)
    var ha = ctx.enqueue_create_host_buffer[dt](M * K)
    var hb = ctx.enqueue_create_host_buffer[dt](K * N)
    ctx.synchronize()
    for i in range(M * K):
        ha[i] = host_a[i]
    for i in range(K * N):
        hb[i] = host_b[i]
    ctx.enqueue_copy(dst_buf=a_buf, src_buf=ha)
    ctx.enqueue_copy(dst_buf=b_buf, src_buf=hb)
    var A = TileTensor(a_buf, g_layout)
    var B = TileTensor(b_buf, g_layout)
    var C = TileTensor(c_buf, g_layout)
    comptime kernel = gpu_matmul[dt, acc]
    ctx.enqueue_function[kernel](
        A,
        B,
        C,
        grid_dim=(ceildiv(N, TILE), ceildiv(M, TILE)),
        block_dim=(TILE, TILE),
    )
    ctx.synchronize()
    var out = List[Scalar[DType.float32]](capacity=M * N)
    with c_buf.map_to_host() as host:
        for i in range(M * N):
            out.append(host[i].cast[DType.float32]())
    return out^


# ------------------------------------------------------------------- driver


def cpu_section() raises:
    print("=== CPU / host numerical section (reference) ===")

    # --- matmul, moderate magnitude -------------------------------------
    var a32 = fill[DType.float32](M * K, 1.0, 11)
    var b32 = fill[DType.float32](K * N, 1.0, 22)
    var ref_c = matmul[DType.float32, DType.float32](a32, b32)
    print("matmul fp32 (acc fp32): min", min_of(ref_c), "max", max_of(ref_c),
          "nan/inf", count_bad(ref_c))
    report("matmul FP32 finite", count_bad(ref_c) == 0)

    var a16 = fill[DType.float16](M * K, 1.0, 11)
    var b16 = fill[DType.float16](K * N, 1.0, 22)
    var c16_native = matmul[DType.float16, DType.float16](a16, b16)
    var c16_acc32 = matmul[DType.float16, DType.float32](a16, b16)
    print("matmul fp16 (acc fp16): nan/inf", count_bad(c16_native),
          "max|err| vs fp32", max_abs_diff(ref_c, c16_native),
          "max rel(>0.1)", max_rel_diff(ref_c, c16_native, 0.1))
    print("matmul fp16 (acc fp32): nan/inf", count_bad(c16_acc32),
          "max|err| vs fp32", max_abs_diff(ref_c, c16_acc32),
          "max rel(>0.1)", max_rel_diff(ref_c, c16_acc32, 0.1))
    report("matmul FP16 finite", count_bad(c16_native) == 0)
    report(
        "FP32 accumulation improves FP16 matmul",
        max_abs_diff(ref_c, c16_acc32) <= max_abs_diff(ref_c, c16_native),
    )

    var abf = fill[DType.bfloat16](M * K, 1.0, 11)
    var bbf = fill[DType.bfloat16](K * N, 1.0, 22)
    var cbf = matmul[DType.bfloat16, DType.float32](abf, bbf)
    print("matmul bf16 (acc fp32): nan/inf", count_bad(cbf),
          "max|err| vs fp32", max_abs_diff(ref_c, cbf),
          "max rel(>0.1)", max_rel_diff(ref_c, cbf, 0.1))
    report("matmul BF16 finite", count_bad(cbf) == 0)

    # --- matmul, large magnitude: the #48 overflow regime ----------------
    # scale 90 gives per-element products ~8.1e3 and K=64 accumulation ~1e5,
    # which exceeds the FP16 finite range (65504) but not FP32.
    var la32 = fill[DType.float32](M * K, 90.0, 11)
    var lb32 = fill[DType.float32](K * N, 90.0, 22)
    var lref = matmul[DType.float32, DType.float32](la32, lb32)
    var la16 = fill[DType.float16](M * K, 90.0, 11)
    var lb16 = fill[DType.float16](K * N, 90.0, 22)
    var l16_native = matmul[DType.float16, DType.float16](la16, lb16)
    var l16_acc32 = matmul[DType.float16, DType.float32](la16, lb16)
    print("large-magnitude matmul: fp32 true max", max_of(lref),
          "(FP16 finite range is 65504)")
    print("  fp16 acc fp16: nan/inf", count_bad(l16_native),
          "zeros", count_zeros(l16_native))
    print("  fp16 acc fp32: nan/inf", count_bad(l16_acc32),
          "zeros", count_zeros(l16_acc32))
    report("FP16 matmul overflow reproduced in the #48 regime",
           count_bad(l16_native) > 0)
    report(
        "FP32 accumulation strictly reduces FP16 overflow count",
        count_bad(l16_acc32) < count_bad(l16_native),
    )
    # Honest limit: FP32 accumulation cannot rescue a value whose TRUE result
    # exceeds the FP16 storage range -- the final downcast still overflows.
    report(
        "FP32 accumulation alone does NOT fully repair overflow (expected)",
        count_bad(l16_acc32) > 0,
    )

    # --- softmax, incl. attention-like score range ----------------------
    var s32 = fill[DType.float32](SEQ, 8.0, 33)
    var sm32 = softmax[DType.float32, DType.float32](s32)
    var s16 = fill[DType.float16](SEQ, 8.0, 33)
    var sm16 = softmax[DType.float16, DType.float16](s16)
    var sm16_a32 = softmax[DType.float16, DType.float32](s16)
    var total = Float32(0.0)
    for i in range(SEQ):
        total += sm32[i]
    print("softmax fp32: sum", total, "max", max_of(sm32),
          "nan/inf", count_bad(sm32))
    print("softmax fp16 (acc fp16): nan/inf", count_bad(sm16),
          "max|err|", max_abs_diff(sm32, sm16))
    print("softmax fp16 (acc fp32): nan/inf", count_bad(sm16_a32),
          "max|err|", max_abs_diff(sm32, sm16_a32))
    report("softmax FP32 sums to 1", abs(total - Float32(1.0)) < Float32(1e-4))
    report("softmax FP16 finite", count_bad(sm16) == 0)

    # extreme attention scores: pre-softmax logits far outside FP16 exp range
    var hot = List[Scalar[DType.float32]](capacity=SEQ)
    var hot16 = List[Scalar[DType.float16]](capacity=SEQ)
    for i in range(SEQ):  # i used below
        var v = Float32(i) * Float32(1.5) - Float32(60.0)
        hot.append(v)
        hot16.append(v.cast[DType.float16]())
    var hsm32 = softmax[DType.float32, DType.float32](hot)
    var hsm16 = softmax[DType.float16, DType.float16](hot16)
    print("extreme-score softmax: fp32 nan/inf", count_bad(hsm32),
          "fp16 nan/inf", count_bad(hsm16),
          "fp16 max|err|", max_abs_diff(hsm32, hsm16))
    report("max-subtracted softmax survives extreme scores in FP16",
           count_bad(hsm16) == 0)

    # --- RMSNorm --------------------------------------------------------
    var x32 = fill[DType.float32](HID, 4.0, 44)
    var rn32 = rmsnorm[DType.float32, DType.float32](x32, 1e-6)
    var x16 = fill[DType.float16](HID, 4.0, 44)
    var rn16 = rmsnorm[DType.float16, DType.float16](x16, 1e-6)
    var rn16_a32 = rmsnorm[DType.float16, DType.float32](x16, 1e-6)
    print("rmsnorm fp32: min", min_of(rn32), "max", max_of(rn32),
          "nan/inf", count_bad(rn32))
    print("rmsnorm fp16 (acc fp16): nan/inf", count_bad(rn16),
          "max|err|", max_abs_diff(rn32, rn16))
    print("rmsnorm fp16 (acc fp32): nan/inf", count_bad(rn16_a32),
          "max|err|", max_abs_diff(rn32, rn16_a32))
    report("rmsnorm FP32 finite", count_bad(rn32) == 0)
    report("rmsnorm FP16 with FP32 accumulation finite",
           count_bad(rn16_a32) == 0)

    # RMSNorm sum-of-squares overflow in pure FP16
    var big32 = fill[DType.float32](HID, 250.0, 44)
    var big16 = fill[DType.float16](HID, 250.0, 44)
    var big_ref = rmsnorm[DType.float32, DType.float32](big32, 1e-6)
    var bad_rn = rmsnorm[DType.float16, DType.float16](big16, 1e-6)
    var good_rn = rmsnorm[DType.float16, DType.float32](big16, 1e-6)
    print("rmsnorm large input (sum-of-squares ~5e6, beyond FP16 range):")
    print("  acc fp16: nan/inf", count_bad(bad_rn), "zeros",
          count_zeros(bad_rn), "max|err| vs fp32", max_abs_diff(big_ref, bad_rn))
    print("  acc fp32: nan/inf", count_bad(good_rn), "zeros",
          count_zeros(good_rn), "max|err| vs fp32",
          max_abs_diff(big_ref, good_rn))
    # The FP16 sum of squares overflows to +Inf, so 1/sqrt(Inf) collapses to 0:
    # every output is finite but WRONG. This is exactly the silent-corruption
    # class of failure seen in Higgs Code2Wav on T4, and a NaN/Inf scan alone
    # would not catch it.
    report(
        "RMSNorm FP16 accumulation corrupts SILENTLY (finite but wrong)",
        count_bad(bad_rn) == 0 and max_abs_diff(big_ref, bad_rn) > Float32(0.5),
    )
    report(
        "RMSNorm collapses to all-zero under FP16 accumulation",
        count_zeros(bad_rn) == len(bad_rn),
    )
    report(
        "FP32 accumulation fully repairs RMSNorm",
        max_abs_diff(big_ref, good_rn) < Float32(0.01),
    )

    # --- SiLU gated MLP -------------------------------------------------
    var g32 = fill[DType.float32](HID, 6.0, 55)
    var u32 = fill[DType.float32](HID, 6.0, 66)
    var mlp32 = silu_gated[DType.float32, DType.float32](g32, u32)
    var g16 = fill[DType.float16](HID, 6.0, 55)
    var u16 = fill[DType.float16](HID, 6.0, 66)
    var mlp16 = silu_gated[DType.float16, DType.float16](g16, u16)
    print("silu-gated fp32: min", min_of(mlp32), "max", max_of(mlp32),
          "nan/inf", count_bad(mlp32))
    print("silu-gated fp16: nan/inf", count_bad(mlp16),
          "max|err|", max_abs_diff(mlp32, mlp16),
          "max rel(>0.1)", max_rel_diff(mlp32, mlp16, 0.1))
    report("silu-gated FP32 finite", count_bad(mlp32) == 0)
    report("silu-gated FP16 finite", count_bad(mlp16) == 0)

    # --- explicit NaN/Inf propagation ------------------------------------
    var poisoned = List[Scalar[DType.float32]](capacity=SEQ)
    for _ in range(SEQ):
        poisoned.append(Float32(0.5))
    poisoned[3] = Float32(0.0) / Float32(0.0)
    poisoned[7] = Float32(1.0) / Float32(0.0)
    var psm = softmax[DType.float32, DType.float32](poisoned)
    print("NaN/Inf propagation: injected 2 bad inputs -> softmax bad count",
          count_bad(psm))
    report("NaN/Inf detected rather than silently swallowed",
           count_bad(psm) > 0)


def gpu_section():
    print("=== GPU / accelerator section ===")
    print("has_accelerator            =", has_accelerator())
    print("has_apple_gpu_accelerator  =", has_apple_gpu_accelerator())
    print("has_nvidia_gpu_accelerator =", has_nvidia_gpu_accelerator())
    print("has_amd_gpu_accelerator    =", has_amd_gpu_accelerator())

    if not has_accelerator():
        print("  [SKIPPED] no accelerator visible to the runtime")
        return

    var ctx: DeviceContext
    try:
        ctx = DeviceContext()
        print("device name =", ctx.name())
        print("device api  =", ctx.api())
    except e:
        print("  [BLOCKED] DeviceContext creation failed:", e)
        return

    # Each step is guarded so a later failure still leaves earlier evidence.
    try:
        var v = run_gpu_vadd(ctx)
        print("gpu vadd c[0] =", v, "(expected 3.0)")
        report("GPU kernel actually executed on device",
               abs(v - Float32(3.0)) < Float32(1e-6))
    except e:
        print("  [BLOCKED] GPU vector-add failed:", e)

    try:
        var g32 = run_gpu_matmul[DType.float32, DType.float32](ctx, 1.0)
        var a32 = fill[DType.float32](M * K, 1.0, 11)
        var b32 = fill[DType.float32](K * N, 1.0, 22)
        var ref_c = matmul[DType.float32, DType.float32](a32, b32)
        print("gpu matmul fp32: nan/inf", count_bad(g32),
              "max|err| vs CPU fp32", max_abs_diff(ref_c, g32))
        report("GPU matmul FP32 matches CPU reference",
               max_abs_diff(ref_c, g32) < Float32(1e-2))
    except e:
        print("  [BLOCKED] GPU matmul fp32 failed:", e)

    try:
        var g16 = run_gpu_matmul[DType.float16, DType.float32](ctx, 1.0)
        var a32b = fill[DType.float32](M * K, 1.0, 11)
        var b32b = fill[DType.float32](K * N, 1.0, 22)
        var ref2 = matmul[DType.float32, DType.float32](a32b, b32b)
        print("gpu matmul fp16 (acc fp32): nan/inf", count_bad(g16),
              "max|err| vs CPU fp32", max_abs_diff(ref2, g16))
        report("GPU matmul FP16 with FP32 accumulation finite",
               count_bad(g16) == 0)
    except e:
        print("  [BLOCKED] GPU matmul fp16 failed:", e)

    try:
        var gbf = run_gpu_matmul[DType.bfloat16, DType.float32](ctx, 1.0)
        print("gpu matmul bf16 (acc fp32): nan/inf", count_bad(gbf))
        report("GPU matmul BF16 supported", count_bad(gbf) == 0)
    except e:
        print("  [SKIPPED/BLOCKED] GPU matmul bf16 unavailable:", e)


def main():
    print("### issue #57 M0 probe: Mojo/MAX hardware + numerical smoke test ###")
    gpu_section()
    print("")
    try:
        cpu_section()
    except e:
        print("  [FAILED] CPU numerical section raised:", e)
    print("")
    print("### end of M0 probe ###")
