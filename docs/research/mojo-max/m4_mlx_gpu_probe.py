"""Parent driver: runs each MLX GPU test case in ITS OWN subprocess (mirrors this project's own
m2_convtranspose1d_prototype.py / m3_decoder_block_prototype.py isolation pattern, needed because
this repo already confirmed one accelerator backend (MAX/Metal) fails via fatal, uncatchable
process abort, not a Python exception -- so a naive in-process try/except could silently hide
later cases). Classifies each case exit as PASSED / FAILED (nonzero exit) / ABORTED (negative
returncode == killed by signal).

Usage: /Users/vedmalex/work/higgs-local-test/.venv-tts/bin/python -u m4_mlx_gpu_probe.py
(run from this directory, or any cwd -- the runner path is resolved relative to this file)
"""
import json
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable  # .venv-tts/bin/python -- has mlx installed
RUNNER = str(Path(__file__).resolve().parent / "m4_mlx_probe_case_runner.py")

CASE_IDS = [
    "ct_stride8_op0",
    "ct_stride5_op1",
    "ct_stride4_op0",
    "ct_stride2_op0",
    "ct_stride3_op1",
    "ct_m3_real_stride5",
    "conv1d_dilation1",
    "conv1d_dilation3",
    "conv1d_dilation9",
    "snake",
]


def run_case(case_id: str) -> dict:
    proc = subprocess.run(
        [PYTHON, "-u", RUNNER, case_id],
        capture_output=True,
        text=True,
        timeout=120,
    )
    rc = proc.returncode
    verdict = "PASSED" if rc == 0 else ("ABORTED" if rc < 0 else "FAILED")
    result_json = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON="):
            try:
                result_json = json.loads(line[len("RESULT_JSON="):])
            except json.JSONDecodeError:
                pass
    if verdict == "PASSED" and result_json is not None:
        if not result_json.get("shape_match", False):
            verdict = "FAILED(shape_mismatch)"
        elif result_json.get("nan_inf", 0) > 0:
            verdict = "FAILED(nan_inf)"
    return {
        "case_id": case_id,
        "returncode": rc,
        "verdict": verdict,
        "result": result_json,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def main():
    print(f"python={PYTHON}")
    all_results = []
    for case_id in CASE_IDS:
        print(f"\n=== running case {case_id} (isolated subprocess) ===", flush=True)
        r = run_case(case_id)
        all_results.append(r)
        print(f"case={case_id} returncode={r['returncode']} verdict={r['verdict']}", flush=True)
        if r["result"] is not None:
            res = r["result"]
            print(
                f"  device={res.get('mlx_default_device')} got_shape={res.get('got_shape')} "
                f"ref_shape={res.get('ref_shape')} max_abs_err={res.get('max_abs_err')} "
                f"nan_inf={res.get('nan_inf')}",
                flush=True,
            )
        else:
            print("  (no RESULT_JSON parsed -- see stdout/stderr tail below)", flush=True)
            if r["stdout_tail"]:
                print("  --- stdout tail ---")
                print(r["stdout_tail"])
            if r["stderr_tail"]:
                print("  --- stderr tail ---")
                print(r["stderr_tail"])

    print("\n\n=== SUMMARY ===")
    for r in all_results:
        res = r["result"] or {}
        print(
            f"{r['case_id']:24s} verdict={r['verdict']:24s} rc={r['returncode']:>4} "
            f"max_abs_err={res.get('max_abs_err')}"
        )

    out_path = str(Path(__file__).resolve().parent / "mlx_gpu_probe_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nfull results written to {out_path}")


if __name__ == "__main__":
    main()
