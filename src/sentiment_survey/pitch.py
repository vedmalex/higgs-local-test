"""Pitch (F0) measurement and pitch-aware pairing for the sentiment survey
(issue #57 follow-up, owner feedback #1).

Why this exists: Higgs generations do not pin voice/speaker identity across
calls (docs/research/audiobook/ measured this directly -- median F0 across
70 segments of one chapter ranges 83.9-203.4 Hz, mean 135.3, stdev 28.6 Hz,
no seed/reference audio passed between calls). That means a "tag" clip and
its "neutral baseline" clip -- or two emotion clips compared against each
other -- can land on opposite sides of the male/female pitch range purely by
chance, independent of any tag or emotion. When that happens, a pair-compare
answer about "which sounds sadder/angrier/etc." mostly measures perceived
voice register, not the tag's effect, and the owner caught exactly this in
practice ("можем ли мы сравнивать однополые голоса?").

This module does NOT reimplement pitch tracking: it reuses the project's own
autocorrelation-based F0 estimator from docs/research/audiobook/
m4_prosody_metrics.py (per-clip median F0 over voiced frames) via a plain
sys.path import -- one estimator, one place, per the owner's instruction.

Threshold derivation (data, not eyeball): every clip referenced by any
loaded task set is measured once, and Otsu's method (1D, two classes) picks
the Hz value that splits the corpus into two groups while minimizing
within-group variance in F0 -- exactly the standard technique for splitting
a distribution into "two natural clusters" (here: the two ends of the
project's own uncontrolled voice range) from actual measured data, with a
single reported number. Two clips are "close enough to compare" if they fall
on the same side of that split.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROSODY_METRICS_PATH = REPO_ROOT / "docs" / "research" / "audiobook" / "m4_prosody_metrics.py"
CACHE_PATH = REPO_ROOT / "output" / "sentiment_survey_results" / "_pitch_cache.json"

# Fallback threshold (Hz) used only when a corpus has too few distinct,
# voiced clips to fit a meaningful two-cluster split (Otsu needs some
# spread to be meaningful) -- e.g. a single hand-written task_sets/*.json
# processed in isolation before the whole-corpus number is available.
# Value: the rough midpoint of the typical modal-male/modal-female speech
# F0 ranges, used ONLY as a last-resort fallback, never as the primary
# derivation (see build_threshold()).
_FALLBACK_THRESHOLD_HZ = 165.0


def _load_prosody_metrics():
    """Import docs/research/audiobook/m4_prosody_metrics.py by path (it is a
    standalone research script, not an installed package)."""
    spec = importlib.util.spec_from_file_location("m4_prosody_metrics", PROSODY_METRICS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {PROSODY_METRICS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_prosody = None


def _prosody_module():
    global _prosody
    if _prosody is None:
        _prosody = _load_prosody_metrics()
    return _prosody


# --------------------------------------------------------------------------- #
# Per-file median F0, cached on disk (autocorrelation over a whole clip is
# not free -- tens to a couple hundred survey clips add up across restarts).
# --------------------------------------------------------------------------- #

def _atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=0)
    os.replace(tmp, path)


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _cache_key(path: Path) -> str:
    return str(path.resolve())


def median_f0_hz(path: Path, cache: dict | None = None) -> float | None:
    """Median F0 (Hz) over voiced frames of one clip, via the project's own
    autocorrelation estimator. None if the file has no reliably voiced
    frames (e.g. pure silence/noise) -- callers must treat that as "unknown
    pitch", never as 0 Hz.

    `cache` is an optional in-memory dict (path -> {mtime, size, f0}) the
    caller can share across many calls in one process/run; when omitted, the
    on-disk cache (CACHE_PATH) is read and written directly (safe for
    occasional/CLI use, a bit wasteful for measuring hundreds of files in a
    loop -- prefer build_f0_index() for that).
    """
    own_cache = cache is None
    if own_cache:
        cache = _load_cache()
    key = _cache_key(path)
    try:
        stat = path.stat()
    except OSError:
        return None
    entry = cache.get(key)
    if entry and entry.get("mtime") == stat.st_mtime and entry.get("size") == stat.st_size:
        return entry.get("f0")

    prosody = _prosody_module()
    result = prosody.analyze(path)
    f0 = result.get("f0_median_hz")
    cache[key] = {"mtime": stat.st_mtime, "size": stat.st_size, "f0": f0}
    if own_cache:
        _atomic_write_json(CACHE_PATH, cache)
    return f0


def build_f0_index(paths: list[Path]) -> dict[str, float | None]:
    """Measure (or fetch from cache) median F0 for many clips in one pass,
    writing the on-disk cache once at the end. Keys are the input paths'
    string form (not resolved) so callers can look values up the same way
    they passed them in."""
    cache = _load_cache()
    index: dict[str, float | None] = {}
    dirty = False
    for path in paths:
        key = _cache_key(path)
        before = cache.get(key)
        f0 = median_f0_hz(path, cache=cache)
        index[str(path)] = f0
        if cache.get(key) is not before:
            dirty = True
    if dirty:
        _atomic_write_json(CACHE_PATH, cache)
    return index


# --------------------------------------------------------------------------- #
# Data-driven threshold (Otsu 1D, two classes) + pairing gate
# --------------------------------------------------------------------------- #

def semitone_diff(f0_a: float, f0_b: float) -> float:
    """Perceptual (log-scale) pitch distance in semitones -- pitch
    perception is roughly logarithmic, so a flat Hz threshold would treat a
    20 Hz gap at the low end (very audible) the same as a 20 Hz gap at the
    high end (barely audible); semitones correct for that."""
    if f0_a <= 0 or f0_b <= 0:
        return float("inf")
    lo, hi = min(f0_a, f0_b), max(f0_a, f0_b)
    import math
    return 12.0 * math.log2(hi / lo)


def otsu_threshold(values: list[float]) -> float | None:
    """Standard 1D Otsu two-class threshold: the value that splits `values`
    into a low and a high group while minimizing the sum of the two groups'
    internal variance (equivalently maximizing between-group variance).
    Returns None if there are fewer than 2 distinct values (no meaningful
    split possible)."""
    uniq = sorted(set(values))
    if len(uniq) < 2:
        return None
    values_sorted = sorted(values)
    n = len(values_sorted)
    best_split = None
    best_score = -1.0
    # Candidate splits: midpoints between consecutive distinct values.
    for i in range(len(uniq) - 1):
        cut = (uniq[i] + uniq[i + 1]) / 2.0
        low = [v for v in values_sorted if v <= cut]
        high = [v for v in values_sorted if v > cut]
        if not low or not high:
            continue
        w_low, w_high = len(low) / n, len(high) / n
        mean_low = sum(low) / len(low)
        mean_high = sum(high) / len(high)
        between_var = w_low * w_high * (mean_low - mean_high) ** 2
        if between_var > best_score:
            best_score = between_var
            best_split = cut
    return best_split


def build_threshold_report(f0_values: list[float]) -> dict:
    """Compute and describe the corpus-wide pitch-pairing threshold from a
    pooled list of median F0 values (one per survey clip, None values
    already dropped by the caller). Returns a dict with the threshold and
    the numbers that justify it, meant to be both used at runtime and
    quoted verbatim in documentation -- the same number both places."""
    values = [v for v in f0_values if v]
    n = len(values)
    if n < 4:
        # Too few points for Otsu to mean anything; fall back to the
        # documented reference number, flagged as such.
        return {
            "method": "fallback (fewer than 4 measured clips in the pooled corpus)",
            "threshold_hz": _FALLBACK_THRESHOLD_HZ,
            "n": n,
        }
    threshold = otsu_threshold(values)
    if threshold is None:
        return {
            "method": "fallback (all measured clips have identical F0)",
            "threshold_hz": _FALLBACK_THRESHOLD_HZ,
            "n": n,
        }
    low = [v for v in values if v <= threshold]
    high = [v for v in values if v > threshold]
    return {
        "method": "otsu-1d (two-class split minimizing within-cluster F0 variance)",
        "threshold_hz": round(threshold, 1),
        "n": n,
        "min_hz": round(min(values), 1),
        "max_hz": round(max(values), 1),
        "mean_hz": round(sum(values) / n, 1),
        "low_cluster_n": len(low),
        "low_cluster_mean_hz": round(sum(low) / len(low), 1) if low else None,
        "high_cluster_n": len(high),
        "high_cluster_mean_hz": round(sum(high) / len(high), 1) if high else None,
    }


def pitch_gate(f0_a: float | None, f0_b: float | None, threshold_hz: float) -> bool:
    """True if two clips are "close enough in pitch" to make a pair-compare
    answer about tag/emotion meaningful: both on the same side of the
    corpus-derived threshold. Unknown pitch (None, e.g. an unvoiced/very
    short clip) is treated as NOT comparable -- we do not guess."""
    if f0_a is None or f0_b is None:
        return False
    return (f0_a <= threshold_hz) == (f0_b <= threshold_hz)


_COMPARISON_TYPES = {"pair_compare", "triple_compare"}


def annotate_pitch_warnings(docs: list[dict]) -> dict:
    """Mutate every pair_compare/triple_compare task across `docs` in place,
    adding a `pitch_warning` key (issue #57 follow-up, owner feedback #1):

      - None if every pair of clips in the task is close enough in pitch
        (same side of the corpus-wide Otsu threshold) to make a
        tag/emotion comparison meaningful.
      - Otherwise a dict {"reason", "pairs": [{"a", "b", "f0_a_hz",
        "f0_b_hz", "semitone_diff"}], "threshold_hz"} describing which
        clip pair(s) are pitch-mismatched (or unmeasurable).

    Tasks are never dropped -- per the owner's explicit instruction, a
    pitch-mismatched pair stays visible and answerable, just honestly
    marked, so the emotion verdict on it can be told apart from a reliable
    one (see server.py's summary handler for how the mismatched bucket is
    kept separate from the graded/differ_pairs statistics).

    Returns the threshold report (see build_threshold_report()) for
    logging/documentation.
    """
    all_paths: list[Path] = []
    seen = set()
    for doc in docs:
        for task in doc.get("tasks", []):
            if task.get("type") not in _COMPARISON_TYPES:
                continue
            for rel in task.get("clips", {}).values():
                abs_path = (REPO_ROOT / rel).resolve()
                if abs_path not in seen:
                    seen.add(abs_path)
                    all_paths.append(abs_path)

    index = build_f0_index(all_paths)
    report = build_threshold_report(list(index.values()))
    threshold = report["threshold_hz"]

    for doc in docs:
        for task in doc.get("tasks", []):
            if task.get("type") not in _COMPARISON_TYPES:
                continue
            clips = task.get("clips", {})
            roles = sorted(clips)
            bad_pairs = []
            for i in range(len(roles)):
                for j in range(i + 1, len(roles)):
                    role_a, role_b = roles[i], roles[j]
                    path_a = str((REPO_ROOT / clips[role_a]).resolve())
                    path_b = str((REPO_ROOT / clips[role_b]).resolve())
                    f0_a, f0_b = index.get(path_a), index.get(path_b)
                    if pitch_gate(f0_a, f0_b, threshold):
                        continue
                    bad_pairs.append({
                        "a": role_a, "b": role_b,
                        "f0_a_hz": f0_a, "f0_b_hz": f0_b,
                        "semitone_diff": (
                            round(semitone_diff(f0_a, f0_b), 1)
                            if f0_a and f0_b else None
                        ),
                    })
            if bad_pairs:
                task["pitch_warning"] = {
                    "reason": (
                        "Клипы в этой паре не близки по высоте основного тона (голос не "
                        "закреплён между генерациями) — сравнение по эмоции/тегу на этой "
                        "паре недостоверно, различие может объясняться просто разным голосом."
                    ),
                    "threshold_hz": threshold,
                    "pairs": bad_pairs,
                }
            else:
                task["pitch_warning"] = None
    return report


# --------------------------------------------------------------------------- #
# CLI: recompute + print a report (used to produce the numbers quoted in
# docs/guides/sentiment_survey_guide.md and to refresh the cache offline).
# --------------------------------------------------------------------------- #

def _iter_task_set_clip_paths():
    """All clip files referenced by every loaded task set (JSON + dynamic),
    without importing server.py (would create a cycle: server imports
    catalog, catalog would need pitch, pitch would need server)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import catalog  # noqa: E402

    seen = set()
    task_sets_dir = Path(__file__).resolve().parent / "task_sets"
    docs = []
    for p in sorted(task_sets_dir.glob("*.json")):
        docs.append(json.loads(p.read_text(encoding="utf-8")))
    docs.extend(catalog.build_all_dynamic_sets())
    for doc in docs:
        for task in doc.get("tasks", []):
            for rel in task.get("clips", {}).values():
                abs_path = (REPO_ROOT / rel).resolve()
                if abs_path not in seen:
                    seen.add(abs_path)
                    yield abs_path


def main():
    paths = list(_iter_task_set_clip_paths())
    index = build_f0_index(paths)
    report = build_threshold_report(list(index.values()))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n{len(paths)} clips measured (cache: {CACHE_PATH}).")


if __name__ == "__main__":
    main()
