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

M3-5 correction (2026-08-24): plain `max_rel_err` was found to be a BROKEN gating metric at
full-block depth. `|got-ref|/|ref|` blows up wherever a reference element lands near zero, and a
25600-element continuous zero-crossing block output essentially guarantees such an element.
Measured across 6 seeds in `m3-block-results.md` (M3-5): `max_abs_err` stays a stable
1.7e-4..2.4e-4 while `max_rel_err` swings 0.014..0.46 purely with how close some element happens
to land to zero; M3-4 saw the same thing (0.036) with zero MAX/GPU involvement, in pure
NumPy/PyTorch. So this module now also computes an `np.allclose`-style COMBINED tolerance,
`|got-ref| <= atol + rtol*|ref|`, which degrades to an absolute check near zero and to a relative
check where `|ref|` is large; that is the gating metric from M3-5 onward. `max_rel_err` is kept
(reported, not gating) and a masked variant is added as a diagnostic. Nothing is removed.

Usage:
    from m3_divergence import compare, DivergenceReport

    report = compare(got, ref)                       # got, ref: array-like, any shape
    report = compare(got, ref, orig_dtype=np.float16) # also check FP16 saturation
    print(report)                                     # human-readable one-liner
    report.max_abs_err, report.nan_count, ...          # individual fields
    report.combined_pass                               # the M3-5-onward gate
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

    # --- M3-5 correction: combined (np.allclose-style) tolerance, the GATING metric ---
    # tol_i = atol_used + rtol_used * |ref_i|;  combined_max_ratio = max_i |err_i| / tol_i.
    # combined_pass is True iff combined_max_ratio <= 1, i.e. every element is inside tolerance.
    rtol_used: float = 0.0
    atol_used: float = 0.0
    atol_scale: Optional[float] = None
    ref_abs_max: float = 0.0
    combined_max_ratio: float = 0.0
    combined_pass: bool = False
    combined_fail_count: int = 0

    # --- diagnostic only: max rel err restricted to non-negligible |ref| ---
    mask_threshold: float = 0.0
    masked_count: int = 0
    max_rel_err_masked: float = 0.0

    def __str__(self) -> str:
        sat = "n/a" if self.saturation_count is None else str(self.saturation_count)
        return (
            f"max|err|={self.max_abs_err:.6g} max_rel_err={self.max_rel_err:.6g} "
            f"combined_ratio={self.combined_max_ratio:.6g} "
            f"(atol={self.atol_used:.6g} rtol={self.rtol_used:.6g}) "
            f"combined_pass={self.combined_pass} fail_n={self.combined_fail_count} "
            f"rel_err_masked(|ref|>={self.mask_threshold:.3g})={self.max_rel_err_masked:.6g} "
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

    def is_healthy_combined(self) -> bool:
        """The M3-5-onward gate: the combined-tolerance check REPLACES the max-abs-err threshold
        of `is_healthy()`, while every E5 structural check (shape, NaN/Inf, the #48 all-zero
        shape) is kept verbatim. `combined_pass` alone is NOT sufficient -- a fully-zeroed `got`
        against a fully-zeroed `ref` would satisfy it trivially."""
        if not self.shape_match:
            return False
        if self.nan_count > 0 or self.inf_count > 0:
            return False
        if not self.combined_pass:
            return False
        if self.exact_zero_count == self.total_elements and self.exact_zero_count_ref != self.total_elements:
            return False
        return True


# Gating defaults, derived in m3-plan.md S5 from M2's and M3-4/M3-5's real measured numbers:
#   rtol 5e-03  -- unchanged from the plan's original relative band (M2 measured 6.45e-04 conv1d
#                  on T4, 9.91e-04 residual unit on M1).
#   atol        -- NOT a global constant: `atol_scale * max|ref|`, so the absolute floor tracks
#                  each tensor's own magnitude instead of silently loosening small-magnitude
#                  tensors. Measured scale-relative errors (max_abs_err / max|ref|):
#                  M2 conv1d 2.37e-06/6.35  = 3.7e-07; M2 residual unit 4.10e-06/13.34 = 3.1e-07;
#                  M3-5 full block 1.89e-04/88 = 2.1e-06 (worst of 6 seeds 2.37e-04/88 = 2.7e-06).
#                  1e-05 leaves ~4x headroom over the deepest measured composite and ~27x over
#                  M2's single ops -- generous but not slack, and it does not relax the ops that
#                  already passed cleanly.
DEFAULT_RTOL = 5e-3
DEFAULT_ATOL_SCALE = 1e-5
# Masking is a DIAGNOSTIC, not the gate (see m3-plan.md S5): |ref| >= mask_scale * max|ref|.
DEFAULT_MASK_SCALE = 1e-3


def compare(
    got: np.ndarray,
    ref: np.ndarray,
    orig_dtype: Optional[np.dtype] = None,
    rel_err_eps: float = 1e-8,
    rtol: float = DEFAULT_RTOL,
    atol: Optional[float] = None,
    atol_scale: Optional[float] = DEFAULT_ATOL_SCALE,
    mask_scale: float = DEFAULT_MASK_SCALE,
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
        Denominator floor for relative error, to avoid divide-by-zero blowups where ref==0. Note
        this floor does NOT rescue `max_rel_err` from the near-zero-denominator artifact M3-4/M3-5
        found empirically -- it only prevents a literal division by zero. That is what the
        combined tolerance below is for.
    rtol, atol, atol_scale : float
        The combined tolerance `|got-ref| <= atol + rtol*|ref|`. If `atol` is given it is used
        verbatim; otherwise `atol = atol_scale * max|ref|` (per-tensor, magnitude-tracking). If
        both are None the absolute term is 0 and the check degenerates to a pure relative one.
    mask_scale : float
        Diagnostic only: `max_rel_err_masked` is the max relative error restricted to elements
        with `|ref| >= mask_scale * max|ref|`.
    """
    got_arr = np.asarray(got)
    ref_arr = np.asarray(ref)

    shape_got = tuple(got_arr.shape)
    shape_ref = tuple(ref_arr.shape)
    shape_match = shape_got == shape_ref

    got64 = got_arr.astype(np.float64)
    ref_abs_max = 0.0
    atol_used = 0.0
    combined_max_ratio = float("inf")
    combined_pass = False
    combined_fail_count = -1
    mask_threshold = 0.0
    masked_count = 0
    max_rel_err_masked = float("inf")
    if shape_match:
        ref64 = ref_arr.astype(np.float64)
        abs_err = np.abs(got64 - ref64)
        abs_ref = np.abs(ref64)
        rel_err = abs_err / (abs_ref + rel_err_eps)
        max_abs_err = float(abs_err.max())
        max_rel_err = float(rel_err.max())

        ref_abs_max = float(abs_ref.max()) if abs_ref.size else 0.0
        if atol is not None:
            atol_used = float(atol)
        elif atol_scale is not None:
            atol_used = float(atol_scale) * ref_abs_max
        tol = atol_used + rtol * abs_ref
        # A zero tolerance (atol=0 and ref==0) must not become 0/0: require an exact match there.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(tol > 0.0, abs_err / np.where(tol > 0.0, tol, 1.0),
                             np.where(abs_err > 0.0, np.inf, 0.0))
        combined_max_ratio = float(ratio.max()) if ratio.size else 0.0
        combined_fail_count = int(np.sum(ratio > 1.0))
        combined_pass = combined_fail_count == 0

        mask_threshold = float(mask_scale) * ref_abs_max
        mask = abs_ref >= mask_threshold
        masked_count = int(np.sum(mask))
        max_rel_err_masked = float(rel_err[mask].max()) if masked_count else 0.0
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
        rtol_used=float(rtol),
        atol_used=atol_used,
        atol_scale=None if atol is not None else (None if atol_scale is None else float(atol_scale)),
        ref_abs_max=ref_abs_max,
        combined_max_ratio=combined_max_ratio,
        combined_pass=combined_pass,
        combined_fail_count=combined_fail_count,
        mask_threshold=mask_threshold,
        masked_count=masked_count,
        max_rel_err_masked=max_rel_err_masked,
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

    # M3-5 correction: the near-zero-denominator case that broke the plain max_rel_err gate.
    # ref has one element at 1.69e-04 amid values of order 88 (M3-5's real output scale); the
    # error there is 3.77e-05 -- ordinary FP32 noise, but rel_err = 0.22.
    ref_nz = np.array([88.0, -50.0, 14.4, -1.69e-04], dtype=np.float64)
    got_nz = ref_nz + np.array([1.0e-04, -5.0e-05, 2.0e-05, 3.77e-05], dtype=np.float64)
    rep_nz = compare(got_nz, ref_nz)
    print("M3-5-shaped near-zero-denominator case:", rep_nz)
    assert rep_nz.max_rel_err > 0.2, "plain max_rel_err must still show the blowup (kept, not hidden)"
    assert rep_nz.combined_pass, "combined tolerance must NOT fail on FP32 noise at a near-zero ref"
    assert rep_nz.atol_used == 1e-5 * 88.0
    assert rep_nz.max_rel_err_masked < 1e-5, "masked rel err must exclude the near-zero element"

    # ... and the combined metric must NOT bless a genuinely wrong element: a 10% error at a
    # large-|ref| element has to fail even though its neighbours are fine.
    got_bad = ref_nz.copy()
    got_bad[0] = 88.0 * 1.10
    rep_bad = compare(got_bad, ref_nz)
    print("genuinely-wrong-element case:", rep_bad)
    assert not rep_bad.combined_pass and not rep_bad.is_healthy_combined()

    # The E5 all-zero trap must still fail under the combined gate, not just under is_healthy().
    assert not compare(got_zeroed, ref).is_healthy_combined()

    print("m3_divergence.py self-checks passed.")
