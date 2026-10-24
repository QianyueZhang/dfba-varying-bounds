"""Run one frozen study in a new workspace, preserving published records."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time

from stage_inputs import stage

ROOT = Path(__file__).resolve().parent
STUDY = "paper1_improvement_2026-10-24"
ANALYTIC = "research_completion/depth_review_2026-10-22/moving_bound_benchmark"
NUMERIC = f"{STUDY}/numerical"
EXTERNAL = f"{STUDY}/external"
CASES = {
    "analytic": {"commands": [[f"{ANALYTIC}/run_benchmark.py"]], "archive": [f"{ANALYTIC}/results"], "group": None},
    "numerical-lp": {"commands": [[f"{NUMERIC}/regularization_experiment.py", "--run-lp"]], "archive": [f"{NUMERIC}/results/lp"], "group": "numerical"},
    "numerical-tight": {"commands": [[f"{NUMERIC}/regularization_experiment.py", "--run-direct", "tight_pair"]], "archive": [f"{NUMERIC}/results/tight_pair"], "group": "numerical"},
    "numerical-small-barrier": {"commands": [[f"{NUMERIC}/regularization_experiment.py", "--run-direct", "fixed_expansion_small_barrier"]], "archive": [f"{NUMERIC}/results/fixed_expansion_small_barrier"], "group": "numerical"},
    "numerical-all": {"commands": [[f"{NUMERIC}/regularization_experiment.py", "--run-lp"], [f"{NUMERIC}/regularization_experiment.py", "--run-direct", "tight_pair"], [f"{NUMERIC}/regularization_experiment.py", "--run-direct", "fixed_expansion_small_barrier"], [f"{NUMERIC}/regularization_experiment.py", "--report"]], "archive": [f"{NUMERIC}/results"], "group": "numerical"},
    "temporal": {"commands": [[f"{STUDY}/scripts/temporal_holdout.py", "--run"]], "archive": [f"{STUDY}/temporal/results.json", f"{STUDY}/temporal/fit_evaluations.jsonl", f"{STUDY}/temporal/observations_predictions.csv", f"{STUDY}/temporal/trajectory_temporal.csv", f"{STUDY}/temporal/trajectory_capacity_prescribed.csv"], "group": "temporal"},
    "temporal-diagnostic": {"commands": [[f"{STUDY}/scripts/temporal_failure_diagnostic.py", "--run"]], "archive": [f"{STUDY}/temporal/failure_diagnostic.json", f"{STUDY}/temporal/trajectory_temporal_prefix.csv", f"{STUDY}/temporal/observations_temporal_prefix.csv", f"{STUDY}/temporal/trajectory_capacity_prescribed.csv", f"{STUDY}/temporal/observations_capacity_prescribed.csv"], "group": "temporal"},
    "external": {"commands": [[f"{EXTERNAL}/cached_scipy_driver.py", "--equivalence"], [f"{EXTERNAL}/cached_scipy_driver.py", "--run"], [f"{EXTERNAL}/report_challenge.py"]], "archive": [f"{EXTERNAL}/runtime_equivalence.json", f"{EXTERNAL}/results.json", f"{EXTERNAL}/results.csv", f"{EXTERNAL}/flux_vectors.npz", f"{EXTERNAL}/summary.json", f"{EXTERNAL}/README.md"], "group": "external"},
    "recalibration": {"commands": [[f"{EXTERNAL}/recalibration/run_recalibration.py"]], "archive": [f"{EXTERNAL}/recalibration/results.json", f"{EXTERNAL}/recalibration/results.csv", f"{EXTERNAL}/recalibration/flux_vectors.npz"], "group": "recalibration"},
}


def runtime():
    return {"python": sys.version, "packages": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "highspy"]}}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=CASES)
    parser.add_argument("--work-dir", required=True, type=Path, help="New directory outside the release; must not exist")
    parser.add_argument("--source-root", type=Path, help="User-held thesis/input tree; required for non-analytic cases")
    parser.add_argument("--prepare-only", action="store_true", help="Stage a fresh copy and print commands without running the study")
    args = parser.parse_args()
    destination = args.work_dir.expanduser().resolve()
    if destination.exists() or destination.is_relative_to(ROOT):
        parser.error("--work-dir must be a new directory outside this release")
    case = CASES[args.case]
    if case["group"] and not args.source_root:
        parser.error("This case requires --source-root containing exact inputs from INPUT_MANIFEST.json")
    for name in ["numpy", "scipy"] + ([] if args.case == "analytic" else ["highspy"]):
        importlib.metadata.version(name)
    if args.case in {"external", "recalibration"} and importlib.metadata.version("scipy") != "1.15.1":
        parser.error("The documented cached postprocessing wrapper requires scipy==1.15.1")
    # Never copy private staged matrices or a virtual environment from this release.
    shutil.copytree(ROOT, destination, ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__", "*.pyc", "*.mat", "*.html", ".DS_Store"))
    staged = stage(args.source_root, destination, [case["group"]]) if case["group"] else None
    archived = []
    for relative in case["archive"]:
        path = destination / relative
        if path.exists():
            saved = destination / "archived_outputs" / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            path.rename(saved)
            archived.append(relative)
    commands = [[sys.executable, *cmd] for cmd in case["commands"]]
    source_hashes = {str(p.relative_to(destination)): sha(p) for p in destination.rglob("*.py") if "archived_outputs" not in p.parts}
    record = {"case": args.case, "fresh_workspace": str(destination), "source_release": str(ROOT), "staged_inputs": staged,
              "archived_copied_outputs": archived, "commands": commands, "status": "prepared", "subprocesses": [],
              "frozen_python_sha256_before": source_hashes, "scope": "Fresh calculation; supplied records remain unchanged. A solver/model failure is retained, not repaired."}
    report_path = destination / "RUN_RECORD.json"
    report_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"workspace": str(destination), "commands": commands, "prepare_only": args.prepare_only}, indent=2), flush=True)
    if args.prepare_only:
        return
    logs = destination / "calculation_logs"
    logs.mkdir()
    code = 0
    for index, command in enumerate(commands):
        tick = time.perf_counter()
        env = dict(os.environ, MPLBACKEND="Agg")
        log_path = logs / f"{index:02d}.log"
        with log_path.open("w") as stream:
            completed = subprocess.run(command, cwd=destination, env=env, stdout=stream, stderr=subprocess.STDOUT)
        record["subprocesses"].append({"command": command, "returncode": completed.returncode, "seconds": time.perf_counter() - tick, "log": str(log_path.relative_to(destination))})
        code = completed.returncode
        record["status"] = "running" if code == 0 else "process_failed"
        report_path.write_text(json.dumps(record, indent=2) + "\n")
        if code:
            break
    record["frozen_python_unchanged"] = all(sha(destination / path) == digest for path, digest in source_hashes.items())
    if args.case == "temporal" and (destination / STUDY / "temporal/results.json").exists():
        outcome = json.loads((destination / STUDY / "temporal/results.json").read_text())
        record["temporal_scientific_outcome"] = outcome.get("forecast")
        if code and outcome.get("forecast", {}).get("complete") is False:
            record["status"] = "recorded_forecast_failure"
    if code == 0:
        record["status"] = "processes_completed"
    if not record["frozen_python_unchanged"]:
        record["status"] = "unexpected_source_change"
        code = 2
    report_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "report": str(report_path), "returncode": code}, indent=2))
    # Preserve nonzero exit status for the known temporal forecast failure.
    raise SystemExit(code)


if __name__ == "__main__":
    main()
