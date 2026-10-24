"""Verify saved-file integrity and reported record completeness; do not rerun fits."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / "paper1_improvement_2026-10-24"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-inputs", action="store_true", help="Also require every external MAT file used by the frozen campaigns")
    args = parser.parse_args()
    errors, omitted, checks = [], [], []

    def expect(label, value):
        checks.append({"check": label, "passed": bool(value)})
        if not value:
            errors.append(label)

    def check_hash(path, expected, label):
        if path.suffix == ".mat" and not path.exists() and not args.require_inputs:
            omitted.append(str(path.relative_to(ROOT)))
            return
        expect(label, path.is_file() and sha(path) == expected)

    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())
    for item in manifest:
        path = ROOT / item["path"]
        expect("release SHA-256: " + item["path"], path.is_file() and sha(path) == item["sha256"])
    numeric = json.loads((STUDY / "numerical/protocol.json").read_text())
    for relative, digest in numeric["source_sha256"].items():
        check_hash(ROOT / relative, digest, "numerical frozen input: " + relative)
    temporal = json.loads((STUDY / "temporal/protocol.json").read_text())
    check_hash(STUDY / "scripts/temporal_holdout.py", temporal["script_sha256"], "temporal frozen script")
    for relative, digest in temporal["sources_sha256"].items():
        check_hash(STUDY / "temporal/source_tree" / relative, digest, "temporal copied source: " + relative)
    diagnostic = json.loads((STUDY / "temporal/failure_diagnostic_protocol.json").read_text())
    check_hash(STUDY / "scripts/temporal_failure_diagnostic.py", diagnostic["script_sha256"], "temporal diagnostic frozen script")
    check_hash(STUDY / "temporal/results.json", diagnostic["primary_results_sha256"], "temporal diagnostic original outcome")
    check_hash(STUDY / "temporal/protocol.json", diagnostic["primary_protocol_sha256"], "temporal diagnostic original protocol")
    external = json.loads((STUDY / "external/protocol.json").read_text())
    check_hash(STUDY / "external/run_challenge.py", external["run_script_sha256"], "external frozen script")
    check_hash(STUDY / "external/prepare_inputs.py", external["prepare_script_sha256"], "external extraction script")
    for relative, digest in external["input_sha256"].items():
        check_hash(STUDY / "external" / relative, digest, "external input: " + relative)
    check_hash(STUDY / "external/observations.json", external["observations_sha256"], "external extracted observations")
    follow = json.loads((STUDY / "external/recalibration/protocol.json").read_text())
    check_hash(STUDY / "external/recalibration/run_recalibration.py", follow["script_sha256"], "recalibration frozen script")
    for relative, digest in follow["input_sha256"].items():
        check_hash(STUDY / "external" / relative, digest, "recalibration input: " + relative)

    nr = json.loads((STUDY / "numerical/results/decomposition.json").read_text())
    expect("numerical decomposition complete", nr["complete"])
    expect("numerical LP checks passed", nr["LP_passed"])
    expect("signed decomposition identity passed", nr["identity_passed"])
    er = json.loads((STUDY / "external/results.json").read_text())
    expect("all 92 external primary records retained", er["primary_lp_count"] == len(er["records"]) == 92)
    expect("all 276 external LP calls retained", er["total_lp_calls"] == 276)
    expect("external primary numerical checks passed", all(row["numerically_valid"] for row in er["records"]))
    fr = json.loads((STUDY / "external/recalibration/results.json").read_text())
    expect("20 follow-up case records and 72 calls retained", len(fr["records"]) == 20 and len(fr["all_solver_calls"]) == 72)
    expect("one-condition maintenance target attained", fr["calibration_success"])
    import numpy as np
    erratum = json.loads((STUDY / "numerical/STEP_COUNT_ERRATUM.json").read_text())
    for row in erratum["records"]:
        record_path = ROOT / row["file"]
        values = np.load(record_path.parent / "accepted_states.npz")
        expect("accepted-step count: " + record_path.parent.name, len(values["time"]) == row["saved_point_count"] and len(values["time"]) - 1 == row["accepted_integration_steps"])
        expect("historical outcome preserved: " + record_path.parent.name, sha(record_path) == row["sha256"])
    expect("prior execution source retained", sha(STUDY / "numerical/reference_execution/regularization_experiment.py") == erratum["producer_before_sha256"])
    expect("prior execution protocol retained", sha(STUDY / "numerical/reference_execution/protocol.json") == erratum["protocol_before_sha256"])
    tr = json.loads((STUDY / "temporal/results.json").read_text())
    expect("negative temporal forecast outcome retained", tr["complete"] is False and tr["forecast"]["complete"] is False and bool(tr["forecast"].get("failure")))
    report = {"passed": not errors, "checks_passed": sum(x["passed"] for x in checks), "checks_total": len(checks),
              "errors": errors, "external_inputs_not_distributed": sorted(set(omitted)),
              "scope": "Saved-byte integrity, frozen-input matching and stated record completeness. No model solve, reproduction timing, independent optimality proof or biological-validation claim.",
              "scientific_outcomes": {"temporal_forecast_complete": tr["forecast"]["complete"], "temporal_forecast_failure": tr["forecast"].get("failure"), "external_primary_cases": er["primary_lp_count"], "followup_calibration_success": fr["calibration_success"]}}
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
