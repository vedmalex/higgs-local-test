"""M3-2 for issue #57 (closes E5 from m1-responsibility-map.md): the reusable per-layer
divergence detector.

Every M2 prototype hand-rolled its own tiny comparison snippet at the bottom of the script
(see `m2_residual_unit_prototype.py`'s final `if __name__ == "__main__":` block: three lines
computing `abs_err`, `rel_err`, and a NaN/Inf count inline). That ad-hoc pattern is exactly what
this module generalizes and replaces for every M3 script going forward.

Per m1-responsibility-map.md S11/E5: a NaN/Inf-only scan reports a fully-zeroed tensor as
"healthy" -- this is the #48 failure shape (an all-zero decoder output that looks "finite" and
passes a naive scan, but is obviously wrong). So exact-zero counting is a first-class field in
the report, not an afterthought bolted on after the fact, and the same goes for saturation
(values at/beyond the *original* dtype's representable ceiling -- e.g. FP16's 65504 -- which a
plain FP64/FP32 comparison after upcast would otherwise hide).

Usage:
    from m3_divergence import compare, DivergenceReport

    report = compare(got, ref)                       # got, ref: array-like, any shape
    report = compare(got, ref, orig_dtype=np.float16) # also check FP16 saturation
    print(report)                                     # human-readable one-liner
    report.max_abs_err, report.nan_count, ...          # individual fields
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Representable-max ("saturation ceiling") for dtypes commonly seen in this project's MAX/Mojo
# work. FP32/FP64 are included for completeness even though saturation at their ceiling is not
# a realistic concern here.
_DTYPE_MAX = {
    np.dtype(np.float16): 65504.0,
    np.dtype(np.float32): float(np.finfo(np.float32).max),
    np.dtype(np.float64): float(np.finfo(np.float64).max),
}


@dataclass
class DivergenceReport:
    """Per-layer/per-tensor divergence report against an FP64 (or otherwise higher-precision)
    reference. All counts are over the full array (all elements), not per-row/per-channel."""

    shape_got: tuple
    shape_ref: tuple
    shape_match: bool
    max_abs_err: float
    max_rel_err: float
    nan_count: int
    inf_count: int
    exact_zero_count: int
    exact_zero_count_ref: int
    saturation_count: Optional[int]
    orig_dtype: Optional[str]
    total_elements: int

    def __str__(self) -> str:
        sat = "n/a" if self.saturation_count is None else str(self.saturation_count)
        return (
            f"max|err|={self.max_abs_err:.6g} max_rel_err={self.max_rel_err:.6g} "
            f"nan={self.nan_count} inf={self.inf_count} "
            f"exact_zero(got)={self.exact_zero_count} exact_zero(ref)={self.exact_zero_count_ref} "
            f"saturation={sat} shape_match={self.shape_match} n={self.total_elements}"
        )

    def is_healthy(self, max_abs_err_threshold: float = 1e-5) -> bool:
        """Convenience gate mirroring the E5 warning literally: NaN/Inf-clean is NOT enough --
        a fully-zeroed `got` (with a non-zero `ref`) must not read as healthy just because it is
        finite everywhere."""
        if not self.shape_match:
            return False
        if self.nan_count > 0 or self.inf_count > 0:
            return False
        if self.max_abs_err > max_abs_err_threshold:
            return False
        # The #48 failure shape: got is (suspiciously) all zero while ref is not.
        if self.exact_zero_count == self.total_elements and self.exact_zero_count_ref != self.total_elements:
            return False
        return True


def compare(
    got: np.ndarray,
    ref: np.ndarray,
    orig_dtype: Optional[np.dtype] = None,
    rel_err_eps: float = 1e-8,
) -> DivergenceReport:
    """Compare `got` (e.g. a MAX graph's output, any numeric dtype) against an FP64 (or other
    high-precision) `ref`.

    Parameters
    ----------
    got : array-like
        The value under test. Upcast to float64 internally for the comparison arithmetic; the
        *original* dtype (pass via `orig_dtype`) is what saturation is checked against, not the
        upcast one.
    ref : array-like
        The higher-precision reference (typically FP64 NumPy).
    orig_dtype : numpy dtype, optional
        The dtype `got` actually had before any upcast (e.g. np.float16). If given and known to
        this module, saturation_count is populated; otherwise it is None (not zero -- "unknown"
        and "zero saturating values" are different facts and must not be conflated).
    rel_err_eps : float
        Denominator floor for relative error, to avoid divide-by-zero blowups where ref==0.
    """
    got_arr = np.asarray(got)
    ref_arr = np.asarray(ref)

    shape_got = tuple(got_arr.shape)
    shape_ref = tuple(ref_arr.shape)
    shape_match = shape_got == shape_ref

    got64 = got_arr.astype(np.float64)
    if shape_match:
        ref64 = ref_arr.astype(np.float64)
        abs_err = np.abs(got64 - ref64)
        rel_err = abs_err / (np.abs(ref64) + rel_err_eps)
        max_abs_err = float(abs_err.max())
        max_rel_err = float(rel_err.max())
    else:
        # Can't do elementwise diff on mismatched shapes -- report the mismatch itself rather
        # than raising or silently broadcasting.
        max_abs_err = float("inf")
        max_rel_err = float("inf")

    nan_count = int(np.sum(np.isnan(got64)))
    inf_count = int(np.sum(np.isinf(got64)))
    exact_zero_count = int(np.sum(got_arr == 0))
    exact_zero_count_ref = int(np.sum(ref_arr == 0))

    saturation_count: Optional[int] = None
    orig_dtype_name: Optional[str] = None
    if orig_dtype is not None:
        dt = np.dtype(orig_dtype)
        orig_dtype_name = dt.name
        ceiling = _DTYPE_MAX.get(dt)
        if ceiling is not None:
            saturation_count = int(np.sum(np.abs(got64) >= ceiling))

    return DivergenceReport(
        shape_got=shape_got,
        shape_ref=shape_ref,
        shape_match=shape_match,
        max_abs_err=max_abs_err,
        max_rel_err=max_rel_err,
        nan_count=nan_count,
        inf_count=inf_count,
        exact_zero_count=exact_zero_count,
        exact_zero_count_ref=exact_zero_count_ref,
        saturation_count=saturation_count,
        orig_dtype=orig_dtype_name,
        total_elements=int(got_arr.size),
    )


if __name__ == "__main__":
    # Self-check with a synthetic case exercising every field, including the E5 zero-tensor trap.
    ref = np.array([1.0, 2.0, 0.0, np.nan_to_num(1e10)], dtype=np.float64)
    got_ok = np.array([1.0000001, 1.9999999, 0.0, 1e10], dtype=np.float32)
    report = compare(got_ok, ref)
    print("synthetic ok case:", report)
    assert report.is_healthy(1e-3)

    got_zeroed = np.zeros_like(ref)
    report_zeroed = compare(got_zeroed, ref)
    print("synthetic #48-shaped all-zero case:", report_zeroed)
    assert not report_zeroed.is_healthy(), "all-zero got vs non-zero ref must NOT read healthy"
    assert report_zeroed.exact_zero_count == 4

    got_fp16_sat = np.array([1.0, 65504.0, 65504.0, 0.0], dtype=np.float64)
    report_sat = compare(got_fp16_sat, ref, orig_dtype=np.float16)
    print("synthetic saturation case:", report_sat)
    assert report_sat.saturation_count == 2

    print("m3_divergence.py self-checks passed.")
