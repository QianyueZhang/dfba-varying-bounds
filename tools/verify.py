"""Run the short numerical verification suite and collect its reports."""

from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "local_verification"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-models", action="store_true",
        help="Also require all exact model inputs and their static LP/KKT checks.",
    )
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    env = os.environ.copy()
    for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        env[key] = "1"
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(OUT / "matplotlib_cache")
    stages = [
        ("installation", ["tests/verify_release.py"] + (["--require-models"] if args.require_models else []),
         OUT / "verification.json"),
        ("moving_bounds", ["experiments/moving_bound_benchmark/run_benchmark.py"],
         ROOT / "experiments/moving_bound_benchmark/results/validation.json"),
        ("neural_statistics", ["experiments/neural_audit_diagnostics/run_diagnostics.py"],
         ROOT / "experiments/neural_audit_diagnostics/results/neural_statistics.json"),
        ("constrained_control", ["experiments/constrained_control/run.py", "verify", "--output", "local_verification/constrained_control.json"],
         OUT / "constrained_control.json"),
        ("terminal_recovery", ["experiments/terminal_recovery/run.py", "verify"],
         ROOT / "experiments/terminal_recovery/local_results/package_verification.json"),
        ("scenario_feedback", ["experiments/scenario_feedback/run.py", "verify"],
         ROOT / "experiments/scenario_feedback/local_results/package_verification.json"),
        ("yeast_assessment", ["experiments/yeast_assessment/run.py", "verify"],
         ROOT / "experiments/yeast_assessment/local_results/package_verification.json"),
        ("sensing_actuation", ["experiments/sensing_actuation/run.py", "verify"],
         ROOT / "experiments/sensing_actuation/local_results/package_verification.json"),
    ]
    results = []
    for name, command, report_path in stages:
        print("Running " + name + "...", flush=True)
        # Remove a stale report so a failed invocation cannot appear to pass.
        report_path.unlink(missing_ok=True)
        started = time.perf_counter()
        log_path = OUT / (name + ".log")
        with log_path.open("w") as log:
            process = subprocess.run(
                [sys.executable, *command], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        row = {
            "stage": name, "command": ["python", *command],
            "exit_code": process.returncode,
            "passed": process.returncode == 0 and report.get("passed") is True,
            "checks_passed": report.get("checks_passed", 0),
            "checks_total": report.get("checks_total", 0),
            "elapsed_seconds": time.perf_counter() - started,
            "report": str(report_path.relative_to(ROOT)),
            "log": str(log_path.relative_to(ROOT)),
            "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest() if report_path.exists() else None,
        }
        results.append(row)
        print(f"{name}: {'PASS' if row['passed'] else 'FAIL'} ({row['checks_passed']}/{row['checks_total']})", flush=True)
    summary = {
        "scope": "Short installation/static-model, analytic moving-bound, saved-prediction and control-study record checks. Full biological simulations and optimisation runs are not included.",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "require_models": args.require_models,
        "passed": all(row["passed"] for row in results),
        "checks_passed": sum(row["checks_passed"] for row in results),
        "checks_total": sum(row["checks_total"] for row in results),
        "stages": results,
    }
    (OUT / "suite.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Report: local_verification/suite.json")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
