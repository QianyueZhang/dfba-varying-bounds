"""Predeclared development comparison of the new engine and independent amounts.

Run --declare before --execute. Existing protocols/results are never overwritten.
No controller optimization or evaluation-campaign replay is performed.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import math
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "audit"
PROTOCOL = HERE / "engine_check_protocol.json"
RESULTS = HERE / "engine_check_results.json"
NOTES = HERE / "ENGINE_CHECK.md"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def cases():
    panel = []
    for family in ("table", "analytic"):
        specifications = [
            ("zero_feed", [1.3, 5., 5., 8.], [0., 0.], None),
            ("constant_low", [1.3, 5., 5., 8.], [.05, .05], None),
            ("early_pulse", [1.3, 5., 5., 8.], [.2, 0.], None),
            ("late_pulse", [1.3, 5., 5., 8.], [0., .2], None),
            ("high_volume", [1.75, 5., 5., 8.], [.01, .01], None),
            ("inhibition", [1.3, 5., 5., 8.], [.05, .05], [180., 260.]),
            ("near_depletion", [.5, .0005, .0005, 5e-9], [0., .05], None),
            ("zero_substrate", [.5, .0005, .0005, 0.], [0., .05], None),
        ]
        for name, initial, feeds, inhibition in specifications:
            panel.append(dict(id=family + "_" + name, family=family,
                              initial_amount_state=initial, feeds=feeds,
                              kappas=[[1., 1.], [.975, 1.025]],
                              inhibition=inhibition))
    return panel


def declare():
    if PROTOCOL.exists() or RESULTS.exists():
        raise RuntimeError("Development check already declared or executed; do not overwrite")
    inputs = ["scripts/engine.py", "vendor/prior_kernel.py",
              "audit/independent_amount_model.py", "data/flux_table_eps1e-06.json",
              "audit/check_engine.py"]
    data = dict(
        version="1.0", declared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope="Fixed development solver check; not held-out controller evidence or an error enclosure.",
        cases=cases(),
        input_sha256={name: digest(ROOT / name) for name in inputs},
        arms={
            "engine": dict(formulation="log biomass/physical substrate amount", method="Dormand-Prince 5(4)", rtol=2e-11, atol=2e-13, max_step_h=.05),
            "amount_original": dict(formulation="physical amounts", method="DOP853", rtol=2e-13, atol=2e-14, max_step_h=.025),
            "amount_refined": dict(formulation="physical amounts", method="DOP853", rtol=5e-14, atol=5e-15, max_step_h=.0125),
        },
        thresholds=dict(amount_absolute_g=2e-7, amount_relative=2e-8,
                        log_ratio_absolute=1e-7, volume_absolute_L=2e-13,
                        substrate_min_g_L=-2e-7, volume_cap_L=1.8,
                        feed_max_L_h=.2, terminal_band=.0075,
                        classification_clearance=2e-5),
        comparisons=["engine versus amount_original", "engine versus amount_refined", "amount_original versus amount_refined"],
        quantity_order=["volume_L", "SC_gDW", "KP_gDW", "glucose_g"],
        checkpoint_rule="Compare both 2h and 4h endpoints; chain each arm from its own endpoint without resets.",
        physical_rule="Positive endpoint biomass, valid feed/volume and substrate minima sampled at accepted solver nodes; not continuous-path certification.",
        classification_rule={
            "fixed_target_ratios": [.8, 1., 1.2],
            "synthetic_target_log_offsets_from_refined_endpoint": [-(.0075+2e-5), -(.0075-2e-5), .0075-2e-5, .0075+2e-5],
            "meaning": "Synthetic target construction is declared before execution and tests numerical band discrimination only; it is not a scientific target-selection procedure.",
            "near_boundary": "Flag fixed targets within 2e-5 of the refined-reference band boundary; still report every label disagreement as a hard failure.",
        },
        failure_rule="Do not loosen thresholds after failure. Record exceptions, missing checkpoints and all failed comparisons. Both original and refined independent arms must pass.",
        timing="Record every interval-call duration. Separate first engine call including possible JIT/cache loading; compare remaining matched calls descriptively. Exclude model construction and file I/O. No online controller speed inference.",
    )
    save(PROTOCOL, data)
    print(json.dumps(dict(protocol=str(PROTOCOL), cases=len(data["cases"]), intervals_per_arm=2*len(data["cases"]), numerical_execution=False)))


def execute():
    if not PROTOCOL.exists():
        raise RuntimeError("Declare protocol first")
    if RESULTS.exists() or NOTES.exists():
        raise RuntimeError("Development results already exist; do not overwrite")
    protocol = json.loads(PROTOCOL.read_text())
    for name, expected in protocol["input_sha256"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("Declared input changed before execution: " + name)
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(HERE))
    import numpy as np
    import scipy
    import numba
    import engine
    import independent_amount_model as independent

    thresholds = protocol["thresholds"]
    if engine.BAND != thresholds["terminal_band"]:
        raise RuntimeError("Engine terminal band differs from declared check")
    report = dict(complete=False, passed=False, protocol_sha256=digest(PROTOCOL),
                  protocol=protocol, cases=[], comparisons=[], failures=[],
                  runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                               numba=numba.__version__, platform=platform.platform()))
    started = time.perf_counter()
    first_engine_call = True

    def fail(code, **context):
        report["failures"].append(dict(code=code, **context))

    def physical(state, minimum, feeds):
        return bool(np.all(np.isfinite(state)) and state[0] > 0
                    and np.all(np.asarray(state[1:3]) > 0)
                    and state[0] <= thresholds["volume_cap_L"]
                    and minimum >= thresholds["substrate_min_g_L"]
                    and all(0 <= u <= thresholds["feed_max_L_h"] for u in feeds))

    for case in protocol["cases"]:
        case_result = dict(case=case, arms={})
        for arm, settings in protocol["arms"].items():
            state = np.array(case["initial_amount_state"], dtype=float)
            initial = state.copy()
            result = dict(checkpoints=[], exceptions=[])
            minimum = float(state[3] / state[0])
            if arm == "engine":
                state = independent.to_log(state)
                arguments = engine.arguments(case["family"], case["inhibition"])
            else:
                model = independent.AmountReactor(case["family"], case["inhibition"])
                independent.RTOL = settings["rtol"]
                independent.ATOL = settings["atol"]
                independent.MAX_STEP = settings["max_step_h"]
            for stage, (feed, kappa) in enumerate(zip(case["feeds"], case["kappas"])):
                call_start = time.perf_counter()
                try:
                    if arm == "engine":
                        state, local_minimum, steps, rejected = engine.flow(
                            state, float(feed), np.array(kappa), arguments,
                            settings["rtol"], settings["atol"], settings["max_step_h"])
                        elapsed = time.perf_counter() - call_start
                        amount_state = independent.from_log(state)
                        work = dict(accepted_steps=int(steps), rejected_steps=int(rejected),
                                    first_engine_call_including_possible_jit=first_engine_call)
                        first_engine_call = False
                    else:
                        output = model.interval(state, float(feed), kappa)
                        elapsed = time.perf_counter() - call_start
                        state = output["amount_state"]
                        amount_state = state.copy()
                        local_minimum = output["min_S"]
                        work = dict(rhs_calls=int(output["rhs_calls"]))
                    minimum = min(minimum, float(local_minimum))
                    valid = physical(amount_state, minimum, case["feeds"][:stage+1])
                    expected_volume = initial[0] + 2.*sum(case["feeds"][:stage+1])
                    volume_error = abs(float(amount_state[0])-float(expected_volume))
                    point = dict(stage=stage, time_h=2*(stage+1), amount_state=amount_state.tolist(),
                                 log_ratio=float(math.log(amount_state[1]/amount_state[2])),
                                 min_substrate_g_L=minimum, physical_valid=valid,
                                 analytic_volume_error_L=volume_error, seconds=elapsed, **work)
                    result["checkpoints"].append(point)
                    if not valid:
                        fail("physical_validity", case=case["id"], arm=arm, stage=stage)
                    if volume_error > thresholds["volume_absolute_L"]:
                        fail("analytic_volume", case=case["id"], arm=arm, stage=stage, error_L=volume_error)
                except Exception as error:
                    item = dict(stage=stage, error_type=type(error).__name__, message=str(error),
                                seconds=time.perf_counter()-call_start)
                    result["exceptions"].append(item)
                    fail("integration_exception", case=case["id"], arm=arm, **item)
                    break
            if len(result["checkpoints"]) != 2:
                fail("missing_checkpoints", case=case["id"], arm=arm,
                     completed=len(result["checkpoints"]), expected=2)
            case_result["arms"][arm] = result
        report["cases"].append(case_result)
        for left, right in (("engine", "amount_original"), ("engine", "amount_refined"), ("amount_original", "amount_refined")):
            left_points = case_result["arms"][left]["checkpoints"]
            right_points = case_result["arms"][right]["checkpoints"]
            reference_points = case_result["arms"]["amount_refined"]["checkpoints"]
            for stage in range(min(len(left_points), len(right_points))):
                a, b = left_points[stage], right_points[stage]
                error = np.abs(np.asarray(a["amount_state"])-np.asarray(b["amount_state"]))
                allowed = thresholds["amount_absolute_g"] + thresholds["amount_relative"]*np.abs(np.asarray(b["amount_state"])[1:])
                bad = np.flatnonzero(error[1:] > allowed)
                z_error = abs(a["log_ratio"]-b["log_ratio"])
                comparison = dict(case=case["id"], left=left, right=right, stage=stage,
                                  component_absolute_error=error.tolist(), allowed_amount_errors=allowed.tolist(),
                                  maximum_amount_error_over_allowance=float(np.max(error[1:]/allowed)),
                                  log_ratio_absolute_error=z_error, classification_checks=[])
                for index in bad:
                    fail("physical_amount_comparison", case=case["id"], left=left, right=right, stage=stage,
                         component=protocol["quantity_order"][index+1], absolute_error=float(error[index+1]),
                         allowed_error=float(allowed[index]))
                if error[0] > thresholds["volume_absolute_L"]:
                    fail("volume_comparison", case=case["id"], left=left, right=right, stage=stage, absolute_error_L=float(error[0]))
                if z_error > thresholds["log_ratio_absolute"]:
                    fail("log_ratio_comparison", case=case["id"], left=left, right=right, stage=stage, absolute_error=z_error)
                if a["physical_valid"] != b["physical_valid"]:
                    fail("physical_classification", case=case["id"], left=left, right=right, stage=stage)
                if stage < len(reference_points):
                    ref_z = reference_points[stage]["log_ratio"]
                    targets = [("fixed", math.log(value)) for value in protocol["classification_rule"]["fixed_target_ratios"]]
                    targets += [("synthetic_band_probe", ref_z+offset) for offset in protocol["classification_rule"]["synthetic_target_log_offsets_from_refined_endpoint"]]
                    for kind, target in targets:
                        label_a = a["physical_valid"] and abs(a["log_ratio"]-target) <= thresholds["terminal_band"]
                        label_b = b["physical_valid"] and abs(b["log_ratio"]-target) <= thresholds["terminal_band"]
                        boundary_distance = abs(abs(ref_z-target)-thresholds["terminal_band"])
                        comparison["classification_checks"].append(dict(kind=kind, target_log_ratio=target,
                            left_label=bool(label_a), right_label=bool(label_b),
                            reference_boundary_distance=boundary_distance,
                            fixed_target_near_boundary=kind == "fixed" and boundary_distance < thresholds["classification_clearance"]))
                        if label_a != label_b:
                            fail("terminal_classification", case=case["id"], left=left, right=right,
                                 stage=stage, kind=kind, target_log_ratio=target)
                report["comparisons"].append(comparison)
        save(RESULTS, report)
        print(case["id"], "completed; cumulative failed records", len(report["failures"]), flush=True)

    report["input_hashes_preserved"] = True
    for name, expected in protocol["input_sha256"].items():
        if digest(ROOT / name) != expected:
            report["input_hashes_preserved"] = False
            fail("input_changed_during_execution", path=name)
    comparisons = report["comparisons"]
    points = {arm: [p for c in report["cases"] for p in c["arms"][arm]["checkpoints"]]
              for arm in protocol["arms"]}
    work = {}
    for arm, arm_points in points.items():
        work[arm] = dict(completed_intervals=len(arm_points), physical_valid_intervals=sum(p["physical_valid"] for p in arm_points),
                         total_seconds=sum(p["seconds"] for p in arm_points),
                         median_interval_seconds=float(np.median([p["seconds"] for p in arm_points])) if arm_points else None)
        if arm == "engine":
            warm = [p for p in arm_points if not p["first_engine_call_including_possible_jit"]]
            first = [p for p in arm_points if p["first_engine_call_including_possible_jit"]]
            work[arm].update(first_call_seconds=first[0]["seconds"] if first else None,
                             later_calls_seconds=sum(p["seconds"] for p in warm),
                             later_calls_median_seconds=float(np.median([p["seconds"] for p in warm])) if warm else None,
                             accepted_steps=sum(p["accepted_steps"] for p in arm_points),
                             rejected_steps=sum(p["rejected_steps"] for p in arm_points))
    report["summary"] = dict(assigned_cases=len(protocol["cases"]), assigned_intervals_per_arm=2*len(protocol["cases"]),
        comparison_records=len(comparisons), failed_records=len(report["failures"]),
        classification_checks=sum(len(c["classification_checks"]) for c in comparisons),
        classification_disagreements=sum(f["code"] == "terminal_classification" for f in report["failures"]),
        fixed_target_near_boundary_records=sum(t["fixed_target_near_boundary"] for c in comparisons for t in c["classification_checks"]),
        maximum_component_absolute_error=np.max([c["component_absolute_error"] for c in comparisons], axis=0).tolist() if comparisons else None,
        maximum_amount_error_over_allowance=max((c["maximum_amount_error_over_allowance"] for c in comparisons), default=None),
        maximum_log_ratio_absolute_error=max((c["log_ratio_absolute_error"] for c in comparisons), default=None),
        per_arm=work)
    report["elapsed_seconds"] = time.perf_counter()-started
    report["complete"] = True
    report["passed"] = not report["failures"]
    save(RESULTS, report)
    lines = ["# Fixed development check of the flow engine", "",
             "Protocol was saved before numerical execution. Original inputs, units and thresholds were retained.", "",
             "Passed all declared checks: **" + str(report["passed"]) + "**.",
             "Assigned cases: %d; intervals per arm: %d; comparison records: %d; failed records: %d." % (len(protocol["cases"]), 2*len(protocol["cases"]), len(comparisons), len(report["failures"])),
             "Terminal classification disagreements: %d of %d comparisons." % (report["summary"]["classification_disagreements"], report["summary"]["classification_checks"]), "",
             "Maximum component errors are ordered volume (L), SC (gDW), KP (gDW), glucose (g): " + str(report["summary"]["maximum_component_absolute_error"]),
             "Maximum log-ratio discrepancy: " + str(report["summary"]["maximum_log_ratio_absolute_error"]), "",
             "## Timing", ""]
    for arm, values in work.items():
        lines.append("- `%s`: %d intervals; %.6g s total; median %s s." % (arm, values["completed_intervals"], values["total_seconds"], values["median_interval_seconds"]))
        if arm == "engine":
            lines.append("  First call, including possible compilation/cache loading: %s s; later-call median: %s s." % (values["first_call_seconds"], values["later_calls_median_seconds"]))
    lines += ["", "## Failed checks", ""]
    lines += ["- " + json.dumps(f) for f in report["failures"]] or ["None."]
    lines += ["", "## Scope", "",
              "This is a fixed development integration check, not a new biological test, an audit of the full recovery library, or a rigorous integration-error enclosure. Physical substrate minima are sampled at solver nodes. Numerical classification probes centered on the refined reference were specified in advance and are not scientific targets. Timing excludes model construction and I/O and establishes no online controller speed claim. A failed check remains failed under the declared tolerances."]
    NOTES.write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(complete=True, passed=report["passed"], summary=report["summary"]), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--declare", action="store_true")
    mode.add_argument("--execute", action="store_true")
    options = parser.parse_args()
    declare() if options.declare else execute()


if __name__ == "__main__":
    main()
