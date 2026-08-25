# M4 batching integration — results (Refs #114)

## 0. The gap this closes

`docs/research/audiobook/m4-batching-results.md` and
`m4-batching-stage-profile-results.md` measured `model.batch_generate()` (3.40x-3.69x
speedup at batch=8, no memory ceiling found up to batch=8) through a standalone script,
[`m4_batching_bench.py`](m4_batching_bench.py) — **not** through the working pipeline.
`generate_segments` in `src/audiobook.py` still called `model.generate()` once per segment.
A 10-hour audiobook therefore still cost the unbatched ~40-60 machine-hours on a real run,
not the ~11.4 hours the batching measurement implied was possible.

## 1. What changed

`generate_segments(..., batch_size: int = 1)` (CLI: `--batch-size`, default **4**):

- `batch_size <= 1` (including the default-less internal case) runs the *exact original*
  code path — one `model.generate()` call per segment, via a new `_generate_single_segment`
  helper that is a pure extraction of the pre-#114 loop body, not a rewrite. This is what
  `--batch-size 1` reproduces byte-for-byte; verified by
  `TestBatchGenerationIntegration.test_batch_size_one_is_byte_for_byte_the_original_unbatched_path`.
- `batch_size > 1`: segments still needing generation (after the existing F6 resume check)
  are grouped into `batch_size`-sized groups and handed to a new `_generate_batch_group`,
  which calls `model.batch_generate()`.

## 2. Decisions on the contested points (per the task's request to be explicit)

- **Segment identity across the batch boundary.** Results are mapped back to segments by
  `BatchGenerationResult.sequence_idx`, never by yield/return order — matching
  `m4_batching_bench.py`'s own convention (`chunk_results.sort(key=lambda r:
  r.sequence_idx)`). In the current `mlx_audio` implementation `sequence_idx` happens to
  equal position (`model.py:633`, `for sequence_idx, text in enumerate(texts)`), but keying
  off it explicitly, not off yield order, survives a future `mlx_audio` revision that
  reorders yields (continuous batching evicts finished rows early, so a future
  implementation could plausibly yield out of order even if today's does not). Verified by
  `test_batch_results_map_to_correct_segments_despite_reversed_yield_order`, which forces a
  fake model to yield in reversed order and checks each segment got audio of the length its
  *own* text implies, not a neighbor's.
- **Manifest write granularity: per segment, not per batch.** `batch_generate()` already
  yields one `BatchGenerationResult` per segment as it finishes decoding (it does not wait
  for the whole batch to write its first result); `_generate_batch_group` calls
  `save_manifest` after each one, exactly like the unbatched path. A kill mid-batch can
  therefore lose at most the segment(s) not yet yielded/validated/written, not the whole
  batch — the same "at most one unit of work lost" guarantee the unbatched path already
  gave per segment. Verified by
  `test_manifest_written_after_each_segment_within_a_batch_not_only_after_whole_batch`
  (counts how many segments are `"done"` at each `save_manifest` call and checks the count
  climbs 1-by-1, not 0-to-N in one jump).
- **Audio-sanity validation: per segment, not per batch.** `_validate_generated_audio` (F6)
  is applied to each decoded row individually inside the batch loop; one implausible-length
  row raises for that segment specifically and does not mark its batch-mates invalid.
  Verified by `test_validation_applied_per_segment_isolates_one_bad_row_in_a_batch`.
- **Retry on failure: whole batch first, then narrowing, never "assume which segment."**
  `batch_generate`'s shared forward pass gives no way to blame one row before the whole
  batch completes (an exception raised mid-decode, or an OOM, is a property of the batch
  call, not a labeled row) — so a batch failure is retried whole, up to `--max-retries`
  times. If it keeps failing at that size, the segments *still not done* (already-succeeded
  ones from an earlier partial attempt are kept, not redone — see `remaining_now`,
  recomputed every attempt) are split in half and each half retried independently,
  recursing down to single-segment `_generate_single_segment`/`model.generate()` calls. This
  isolates the actually-bad segment(s) without discarding batch-mates that were fine, and
  doubles as automatic degradation toward a smaller effective batch size if the failure is
  memory pressure rather than a bad segment. Verified by
  `test_whole_batch_failure_degrades_and_still_completes_via_fallback` (whole-batch failure
  for many attempts, still finishes via the single-segment fallback) and
  `test_validation_applied_per_segment_isolates_one_bad_row_in_a_batch` (one permanently-bad
  row in `batch_generate` isolated down to a `model.generate()` fallback that succeeds,
  while its batch-mates are unaffected).
- **Resume after interruption mid-batch.** A segment left `"in_progress"` (or reset to
  `"pending"`) by a killed run falls through to (re)generation exactly like `"pending"` did
  before batching — no new resume-state handling was needed, because the pre-existing
  status-based loop already treated any non-`"done"` status as "needs generation." Verified
  by `test_resume_after_interruption_mid_batch_only_regenerates_missing_segment`, which
  deletes one segment's WAV out from under an otherwise-complete manifest and confirms only
  that one segment is re-requested from the model (via the single-segment path, since a lone
  leftover segment gains nothing from batching).
- **Memory hygiene (`--clear-cache-every`).** Left counted in *segments actually generated*
  (a resumed/skipped segment still does not count, matching pre-#114 behavior), not in
  batches — this keeps the cadence's meaning ("clear after N units of real GPU work")
  unchanged regardless of `--batch-size`, rather than requiring the operator to recompute
  N/batch_size by hand.
- **Memory ceiling.** No production-side detector for a specific OOM signature was added —
  the whole-batch-retry-then-halve-then-single-segment fallback above already produces
  graceful degradation for *any* batch-level failure, memory pressure included, without
  needing to distinguish "memory error" from "other error" (mlx_audio does not raise a
  distinguishable OOM exception type to key off). Default `--batch-size` is **4**, not the
  measured-best 8, specifically because the existing measurements
  (`m4-batching-results.md`, `m4-batching-stage-profile-results.md`) covered only dozens of
  segments per run, never the hundreds a real chapter has, and found no ceiling only within
  that smaller range — 4 leaves headroom before hitting whatever a several-hundred-segment
  run's actual ceiling turns out to be. `--batch-size 8` reproduces the exact measured
  configuration for anyone who wants to push to the previously-confirmed-safe depth.

## 3. Unit tests (synthetic, no GPU) — `TestBatchGenerationIntegration`

Added to `tests/test_audiobook.py`, 6 new tests, all passing:

- `test_batch_results_map_to_correct_segments_despite_reversed_yield_order`
- `test_batch_size_one_is_byte_for_byte_the_original_unbatched_path`
- `test_resume_after_interruption_mid_batch_only_regenerates_missing_segment`
- `test_validation_applied_per_segment_isolates_one_bad_row_in_a_batch`
- `test_manifest_written_after_each_segment_within_a_batch_not_only_after_whole_batch`
- `test_whole_batch_failure_degrades_and_still_completes_via_fallback`

Full suite result (`.venv-tts/bin/python3 -m unittest tests.test_audiobook -v`):

```
Ran 78 tests in 0.1s

OK
```

(72 pre-existing + 6 new; all pre-existing tests pass unchanged, confirming `--batch-size 1`
did not disturb any prior behavior.)

## 4. Real hardware before/after measurement — STATUS: deferred, will ride the real chapter run

**Decision (2026-08-25, owner call):** do not hold this PR on a separate synthetic
before/after benchmark. `_segment_hash_input` (`speaker + "\x1f" + text`) does not depend on
`batch_size` in any way, and `batch_size` is deliberately **not** part of the manifest header
(`build_manifest` only stores `model`/`max_chars`/`tag_scope`) or the resume mismatch check
in `load_or_create_manifest` — confirmed by reading both functions and by a direct
reproduction: a manifest fully generated at `--batch-size 1`, reloaded and re-run at
`--batch-size 8`, raises no mismatch and re-validates every segment as already `"done"`
without recontacting the model. This means the in-progress full-chapter baseline run
(`/private/tmp/higgs-wt-114-full-chapter`, `output/chapter-114-e0/manifest.json`, unbatched,
29/70 segments done at the time this decision was made) can simply be resumed with
`--batch-size 8` in the same `--output-dir` once this PR lands there: the 29 already-done
segments are reused from the manifest untouched, and the remaining ~41 are generated through
the batched path — producing, in one real run, both the unbatched baseline (the first 29,
already measured) and the batched behavior (the rest, on the real chapter's actual segment
lengths and count) on directly comparable material, which is more informative than a
separate synthetic 20-segment run would have been. That re-run is tracked as follow-up work
outside this PR (it runs against the full-chapter worktree, not this one).

The fixture and exact commands below remain valid and are kept for anyone who wants an
isolated, from-scratch comparison instead of (or in addition to) the resumed-chapter
approach:

```bash
# baseline -- exact pre-#114 behavior
.venv-tts/bin/python3 src/audiobook.py \
  --text-file docs/research/audiobook/m4_batching_integration_segments_ru.txt \
  --output-dir output/m4_batching_integration/batch1 \
  --max-chars 90 --batch-size 1 --assemble

# batched
.venv-tts/bin/python3 src/audiobook.py \
  --text-file docs/research/audiobook/m4_batching_integration_segments_ru.txt \
  --output-dir output/m4_batching_integration/batch4 \
  --max-chars 90 --batch-size 4 --assemble
```

`--max-chars 90` was chosen (and confirmed via `--dry-run`) to produce exactly 20 chunks
from [`m4_batching_integration_segments_ru.txt`](m4_batching_integration_segments_ru.txt),
one per sentence.

To report, whichever route is used: aggregate RTF (total wall / total audio duration) per
`--batch-size`, and the three named memory quantities per this project's convention --
`peak_mlx (GiB)` (`mx.get_peak_memory()`), `peak_footprint (GiB)` (`/usr/bin/time -l`'s
"peak memory footprint" line), and `weights_on_disk (GiB)` (the checkpoint's on-disk size) --
**never `ru_maxrss`**. Expectation, per the existing bench measurements: aggregate RTF
should improve by roughly 3x at `--batch-size 4`-8 vs. `--batch-size 1`. If it does not, that
is itself the reportable result — the per-segment manifest/validation/retry bookkeeping
wrapped around each `batch_generate()` call is new overhead the standalone bench script
never paid, and if it turns out to eat most of the speedup, that must be published as-is and
investigated, not hidden.
