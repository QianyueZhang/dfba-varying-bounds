#!/usr/bin/env python3
"""Direct finite-barrier integration of a specified feed policy.

Requires the four exact yeast model inputs from data/model_inputs.json.
The intracellular equations are the same as those used for the response table.
These checks assess numerical consistency, not independent biological validity.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[3]
DEFAULTS = {
    "horizon_h": 14.0,
    "inlet_glucose_g_L": 100.0,
    "feed_bounds_L_h": [0.0, 0.2],
    "volume_cap_L": 1.8,
    "volume_acceptance_tolerance_L": 1e-8,
    "ratio_log_tolerance": math.log(1.01),
    "optimization_log_tolerance": 0.009,
    "state_negative_tolerance": 1e-8,
    "boundary_ramp_g_L": 1e-5,
    "epsilon": 1e-6,
    "mu": 1e-6,
    "rtol": 5e-8,
    "atol": 5e-11,
    "max_step_h": 0.25,
    "KKT_tolerance": 1e-11,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_backend(root: Path = ROOT):
    """Load the intracellular solver definitions without running their script entry points."""
    namespace: dict[str, Any] = {"ROOT": root, "__name__": "_readonly_kkt_backend"}
    source_records = []
    for name in ("flux_model.py", "primal_dual.py"):
        path = root / "research_completion" / "scripts" / name
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        nodes = [
            node for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef))
            and not (isinstance(node, ast.ImportFrom) and node.module == "flux_model")
        ]
        namespace["__file__"] = str(path)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        source_records.append({"path": str(path.relative_to(root)), "sha256": _sha256(path)})
    return namespace["FluxModel"], namespace["solve"], source_records


class DirectBackend:
    """Warm-start KKT solves, recording successful and failed attempts."""

    def __init__(self, settings: dict[str, Any], root: Path = ROOT):
        started = time.perf_counter()
        model_type, self.solve, self.sources = load_backend(root)
        self.models = [model_type(name, settings["epsilon"]) for name in ("SC", "KP")]
        self.settings = settings
        self.warm: list[Any] = [None, None]
        self.calls = 0
        self.successes = 0
        self.retries = 0
        self.solve_seconds = 0.0
        self.failed_attempts: list[dict[str, Any]] = []
        self.species_stats = {
            name: {
                "successful_solves": 0,
                "newton_steps": 0,
                "max_equality_residual": 0.0,
                "max_stationarity_residual": 0.0,
                "max_primal_dual_residual": 0.0,
                "minimum_slack": math.inf,
                "growth_min": math.inf, "growth_max": -math.inf,
                "glucose_flux_min": math.inf, "glucose_flux_max": -math.inf,
                "solve_seconds": 0.0,
            } for name in ("SC", "KP")
        }
        self.model_load_seconds = time.perf_counter() - started
        self.model_provenance = []
        for model in self.models:
            record = dict(model.provenance)
            raw_path = root / record["raw_bounds_source"]
            record["raw_bounds_sha256"] = _sha256(raw_path)
            self.model_provenance.append(record)

    def rates(self, substrate: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Return raw rates; depletion scaling belongs in reactor equations."""
        s_use = max(float(substrate), 0.0)
        rates, details = [], []
        for i, model in enumerate(self.models):
            warm = self.warm[i]
            for attempt in range(2 if warm is not None else 1):
                started = time.perf_counter()
                self.calls += 1
                try:
                    result = self.solve(
                        model, s_use, mu=self.settings["mu"],
                        tol=self.settings["KKT_tolerance"],
                        warm=warm if attempt == 0 else None,
                    )
                except RuntimeError as error:
                    elapsed = time.perf_counter() - started
                    self.solve_seconds += elapsed
                    self.species_stats[model.name]["solve_seconds"] += elapsed
                    self.failed_attempts.append({
                        "species": model.name, "substrate_g_L": s_use,
                        "warm_start": warm is not None and attempt == 0,
                        "error": str(error), "seconds": elapsed,
                    })
                    if warm is None or attempt == 1:
                        raise
                    self.retries += 1
                    continue
                elapsed = time.perf_counter() - started
                self.solve_seconds += elapsed
                self.species_stats[model.name]["solve_seconds"] += elapsed
                break
            self.warm[i] = result[:4]
            d = result[-1]
            if not all(np.isfinite(d[k]) for k in (
                "growth", "glucose_flux", "equality_residual",
                "stationarity_residual", "primal_dual_residual", "minimum_slack"
            )):
                raise RuntimeError(f"Non-finite KKT output: {model.name}, S={s_use}")
            self.successes += 1
            stats = self.species_stats[model.name]
            stats["successful_solves"] += 1
            stats["newton_steps"] += d["newton_steps"]
            for key in ("equality_residual", "stationarity_residual", "primal_dual_residual"):
                stats["max_" + key] = max(stats["max_" + key], d[key])
            stats["minimum_slack"] = min(stats["minimum_slack"], d["minimum_slack"])
            for key in ("growth", "glucose_flux"):
                stats[key + "_min"] = min(stats[key + "_min"], d[key])
                stats[key + "_max"] = max(stats[key + "_max"], d[key])
            rates.append([d["growth"], d["glucose_flux"]])
            details.append(d)
        return np.asarray(rates), details

    def report(self) -> dict[str, Any]:
        return {
            "backend_label": "same original finite-barrier primal_dual.solve at every RHS evaluation",
            "independent_backend": False,
            "source_files": self.sources,
            "model_provenance": self.model_provenance,
            "KKT_solve_attempts": self.calls,
            "KKT_successful_solves": self.successes,
            "cold_retries_after_failed_warm_start": self.retries,
            "failed_attempts": self.failed_attempts,
            "KKT_solve_seconds": self.solve_seconds,
            "model_load_seconds": self.model_load_seconds,
            "species": self.species_stats,
            "max_equality_residual": max(x["max_equality_residual"] for x in self.species_stats.values()),
            "max_stationarity_residual": max(x["max_stationarity_residual"] for x in self.species_stats.values()),
            "max_primal_dual_residual": max(x["max_primal_dual_residual"] for x in self.species_stats.values()),
            "minimum_slack": min(x["minimum_slack"] for x in self.species_stats.values()),
            "tolerance_interpretation": (
                "KKT_tolerance is the stopping tolerance on the maximum primal-dual, equality, "
                "and complementarity residual. Explicit barrier stationarity is separately "
                "recomputed and may be larger near narrow bounds; it is not asserted to meet that tolerance."
            ),
        }


def _validated_input(payload: dict[str, Any]):
    settings = dict(DEFAULTS)
    overrides = payload.get("settings", {})
    unknown = set(overrides) - set(settings)
    if unknown:
        raise ValueError(f"Unknown settings: {sorted(unknown)}")
    settings.update(overrides)
    controls_raw = payload.get("controls", payload.get("control", payload.get("policy_controls")))
    if controls_raw is None and isinstance(payload.get("policy"), dict):
        controls_raw = payload["policy"].get("controls", payload["policy"].get("control"))
    controls = np.asarray(controls_raw, dtype=float)
    initial = np.asarray(payload["initial"], dtype=float)
    target = float(payload["target_ratio"])
    if controls.shape != (7,) or not np.all(np.isfinite(controls)):
        raise ValueError("controls must contain exactly seven finite feed rates")
    if initial.shape != (4,) or not np.all(np.isfinite(initial)):
        raise ValueError("initial must be finite [volume, X_SC, X_KP, substrate]")
    if initial[0] <= 0 or initial[1] <= 0 or initial[2] <= 0 or initial[3] < 0:
        raise ValueError("Initial volume and both biomass concentrations must be positive; S >= 0")
    if not np.isfinite(target) or target <= 0:
        raise ValueError("target_ratio must be finite and positive")
    for name, value in settings.items():
        if name == "feed_bounds_L_h":
            if len(value) != 2 or not np.all(np.isfinite(value)) or value[0] > value[1]:
                raise ValueError("feed_bounds_L_h must be [lower, upper]")
        elif not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    endpoints = initial[0] + settings["horizon_h"] / 7 * np.cumsum(controls)
    if np.min(endpoints) <= 0:
        raise ValueError("Policy yields nonpositive reactor volume")
    return settings, controls, initial, target


def verify_policy(payload: dict[str, Any], root: Path = ROOT, progress: bool = True):
    """Return JSON-serializable report and an array of accepted trajectory nodes."""
    started = time.perf_counter()
    settings, controls, initial, target = _validated_input(payload)
    backend = DirectBackend(settings, root)
    interval_h = settings["horizon_h"] / 7
    # Amount formulation analytically cancels the dilution terms.
    amounts = initial[0] * initial[1:]
    rhs_count = 0
    rhs_seconds = 0.0
    rhs_min_concentration = initial[1:].copy()
    rhs_min_amount = amounts.copy()
    rhs_min_volume = float(initial[0])
    rhs_max_volume = float(initial[0])
    effective_rate_min = np.full((2, 2), np.inf)
    effective_rate_max = np.full((2, 2), -np.inf)
    segments = []
    trajectory = [[0.0, *initial, *amounts, 0.0, float(controls[0])]]
    integration_started = time.perf_counter()
    for j, feed in enumerate(controls):
        t0, t1 = j * interval_h, (j + 1) * interval_h
        v0 = float(initial[0] + interval_h * np.sum(controls[:j]))

        def rhs(t, mass):
            nonlocal rhs_count, rhs_seconds, rhs_min_volume, rhs_max_volume
            nonlocal rhs_min_concentration, rhs_min_amount, effective_rate_min, effective_rate_max
            rhs_started = time.perf_counter()
            volume = v0 + float(feed) * (t - t0)
            concentration = mass / volume
            rhs_min_concentration = np.minimum(rhs_min_concentration, concentration)
            rhs_min_amount = np.minimum(rhs_min_amount, mass)
            rhs_min_volume = min(rhs_min_volume, float(volume))
            rhs_max_volume = max(rhs_max_volume, float(volume))
            rates, _ = backend.rates(float(concentration[2]))
            factor = min(1.0, max(float(concentration[2]), 0.0) / settings["boundary_ramp_g_L"])
            rates = rates * factor
            effective_rate_min = np.minimum(effective_rate_min, rates)
            effective_rate_max = np.maximum(effective_rate_max, rates)
            derivative = np.array([
                rates[0, 0] * mass[0],
                rates[1, 0] * mass[1],
                float(feed) * settings["inlet_glucose_g_L"]
                + 0.18 * (rates[0, 1] * mass[0] + rates[1, 1] * mass[1]),
            ])
            rhs_count += 1
            rhs_seconds += time.perf_counter() - rhs_started
            return derivative

        segment_started = time.perf_counter()
        solution = solve_ivp(
            rhs, (t0, t1), amounts, method="DOP853",
            rtol=settings["rtol"], atol=settings["atol"], max_step=settings["max_step_h"],
        )
        segment_seconds = time.perf_counter() - segment_started
        if not solution.success:
            raise RuntimeError(f"DOP853 failed at interval {j}: {solution.message}")
        for t, mass in zip(solution.t[1:], solution.y.T[1:]):
            volume = v0 + float(feed) * (float(t) - t0)
            trajectory.append([float(t), volume, *(mass / volume), *mass, float(j), float(feed)])
        amounts = solution.y[:, -1]
        segments.append({
            "interval_index": j, "time_start_h": t0, "time_end_h": t1,
            "feed_L_h": float(feed), "rhs_evaluations": solution.nfev,
            "accepted_steps": len(solution.t) - 1, "seconds": segment_seconds,
        })
        if progress:
            print(f"direct KKT: interval {j + 1}/7; RHS={rhs_count}; seconds={time.perf_counter() - integration_started:.1f}", flush=True)
    integration_seconds = time.perf_counter() - integration_started
    trajectory = np.asarray(trajectory)
    terminal = trajectory[-1, 1:5]
    log_ratio = math.log(terminal[1] / terminal[2])
    log_error = log_ratio - math.log(target)
    ratio_margin = settings["ratio_log_tolerance"] - abs(log_error)
    optimization_margin = settings["optimization_log_tolerance"] - abs(log_error)
    volume_margin = settings["volume_cap_L"] - float(np.max(trajectory[:, 1]))
    lower, upper = settings["feed_bounds_L_h"]
    rate_lower_margin = float(np.min(controls) - lower)
    rate_upper_margin = float(upper - np.max(controls))
    accepted_min = np.min(trajectory[:, 2:5], axis=0)
    accepted_amount_min = np.min(trajectory[:, 5:8], axis=0)
    minstate_margin = float(np.min(accepted_min) + settings["state_negative_tolerance"])
    diagnostics = backend.report()
    kkt_ok = (
        diagnostics["max_primal_dual_residual"] < settings["KKT_tolerance"]
        and diagnostics["minimum_slack"] > 0
    )
    # Stage minima are reported diagnostically, not silently clipped in outputs.
    # Flux evaluation uses max(S,0) exactly as the original depletion convention.
    constraint_checks = {
        "ratio_nominal_pass": ratio_margin >= 0,
        "ratio_nominal_log_margin": ratio_margin,
        "ratio_optimization_pass": optimization_margin >= 0,
        "ratio_optimization_log_margin": optimization_margin,
        "volume_pass": volume_margin >= -settings["volume_acceptance_tolerance_L"],
        "volume_strict_pass": volume_margin >= 0,
        "volume_margin_L": volume_margin,
        "volume_acceptance_margin_L": volume_margin + settings["volume_acceptance_tolerance_L"],
        "feed_bounds_pass": rate_lower_margin >= -1e-12 and rate_upper_margin >= -1e-12,
        "feed_lower_margin_L_h": rate_lower_margin,
        "feed_upper_margin_L_h": rate_upper_margin,
        "accepted_state_nonnegative_within_tolerance": minstate_margin >= 0,
        "accepted_minstate_margin": minstate_margin,
        "minimum_accepted_concentrations_SC_KP_S": accepted_min.tolist(),
        "minimum_accepted_amounts_SC_KP_S": accepted_amount_min.tolist(),
        "minimum_rhs_stage_concentrations_SC_KP_S": rhs_min_concentration.tolist(),
        "minimum_rhs_stage_amounts_SC_KP_S": rhs_min_amount.tolist(),
        "rhs_stage_nonnegative_within_tolerance": bool(np.min(rhs_min_concentration) >= -settings["state_negative_tolerance"]),
        "volume_analytic_L": float(initial[0] + interval_h * np.sum(controls)),
        "terminal_volume_analytic_error_L": float(terminal[0] - initial[0] - interval_h * np.sum(controls)),
        "minimum_volume_L": float(np.min(trajectory[:, 1])),
        "maximum_volume_L": float(np.max(trajectory[:, 1])),
        "rhs_stage_minimum_volume_L": rhs_min_volume,
        "rhs_stage_maximum_volume_L": rhs_max_volume,
        "primal_dual_stopping_and_strict_slack_pass": kkt_ok,
    }
    nominal_feasible = all(constraint_checks[k] for k in (
        "ratio_nominal_pass", "volume_pass", "feed_bounds_pass",
        "accepted_state_nonnegative_within_tolerance", "primal_dual_stopping_and_strict_slack_pass",
    ))
    report = {
        "schema_version": 1,
        "status": "completed",
        "condition_id": payload.get("condition_id"),
        "method": payload.get("method"),
        "seed": payload.get("seed"),
        "controls_L_h": controls.tolist(),
        "initial_V_XSC_XKP_S": initial.tolist(),
        "target_ratio": target,
        "settings": settings,
        "numerical_formulation": {
            "integrated_state": ["M_SC_gDW", "M_KP_gDW", "VS_g"],
            "volume": "analytic piecewise-linear volume; control changes exactly at interval boundaries",
            "glucose_flux_to_mass_factor": 0.18,
            "depletion_convention": "rates times min(1,max(S,0)/boundary_ramp_g_L); KKT evaluated at max(S,0)",
            "trajectory_sampling": "all accepted DOP853 step endpoints plus initial state",
            "atol_applies_to": "amounts, not concentrations",
        },
        "terminal": terminal.tolist(),
        "terminal_names": ["V_L", "X_SC_gDW_L", "X_KP_gDW_L", "S_g_L"],
        "terminal_amounts": amounts.tolist(),
        "terminal_biomass_gDW": float(amounts[0] + amounts[1]),
        "terminal_ratio_SC_KP": float(terminal[1] / terminal[2]),
        "terminal_log_ratio": log_ratio,
        "target_log_ratio": math.log(target),
        "signed_log_ratio_target_error": log_error,
        "absolute_log_ratio_target_error": abs(log_error),
        "nominal_feasible": nominal_feasible,
        "constraint_checks": constraint_checks,
        "effective_rates": {
            "rows": ["SC", "KP"], "columns": ["growth_h_inverse", "glucose_flux_mmol_gDW_h"],
            "minimum_over_rhs_stages": effective_rate_min.tolist(),
            "maximum_over_rhs_stages": effective_rate_max.tolist(),
        },
        "KKT_diagnostics": diagnostics,
        "rhs_evaluations": rhs_count,
        "rhs_seconds": rhs_seconds,
        "integration_seconds": integration_seconds,
        "wall_seconds": time.perf_counter() - started,
        "segments": segments,
        "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
        "interpretation": (
            "Direct trajectory verification under the same finite-barrier model used for interpolation. "
            "Nominal feasibility uses accepted-state minima and reported tolerances; all RHS-stage minima "
            "are also exposed. This is not an independent biological validation or proof of optimality."
        ),
    }
    return report, trajectory


def _write_json(path: Path, content: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Policy JSON")
    parser.add_argument("--output", type=Path, required=True, help="Result JSON; companion trajectory CSV is written alongside")
    parser.add_argument("--check-point", type=float, help="Check both KKT models at one substrate value; no trajectory")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root containing prepared model inputs")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.check_point is not None:
        if args.input is not None:
            parser.error("--input and --check-point are mutually exclusive")
        if not np.isfinite(args.check_point) or args.check_point < 0:
            parser.error("--check-point must be finite and nonnegative")
        started = time.perf_counter()
        backend = DirectBackend(DEFAULTS, args.root)
        rates, details = backend.rates(args.check_point)
        report = {
            "status": "point_check_completed", "substrate_g_L": args.check_point,
            "settings": DEFAULTS, "rates_SC_KP_growth_glucose": rates.tolist(),
            "details_SC_KP": details, "KKT_diagnostics": backend.report(),
            "wall_seconds": time.perf_counter() - started,
        }
        _write_json(args.output, report)
    else:
        if args.input is None:
            parser.error("--input is required unless --check-point is provided")
        payload = json.loads(args.input.read_text())
        report, trajectory = verify_policy(payload, args.root, progress=not args.quiet)
        report["input_path"] = str(args.input.resolve())
        report["input_sha256"] = _sha256(args.input)
        csv_path = args.output.with_name(args.output.stem + "_trajectory.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_h", "V_L", "X_SC_gDW_L", "X_KP_gDW_L", "S_g_L", "M_SC_gDW", "M_KP_gDW", "VS_g", "interval_index", "feed_L_h"])
            writer.writerows(trajectory)
        report["trajectory_csv"] = str(csv_path.resolve())
        report["trajectory_sha256"] = _sha256(csv_path)
        _write_json(args.output, report)
    print(json.dumps({"output": str(args.output.resolve()), "status": report["status"], "wall_seconds": report["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
