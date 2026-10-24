#!/usr/bin/env python3
"""Frozen public-data analysis; no access to yeast campaign outcomes or models."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from scipy.optimize import least_squares
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "audit" / "mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def nested(value, keys):
    for key in keys:
        value = value[key] if isinstance(value, dict) else getattr(value, key)
    return value


def read_data(protocol):
    source = STUDY / "literature_sources" / "Kumar2022_source_data.zip"
    assert sha(source) == protocol["source"]["sha256"]
    archive = zipfile.ZipFile(source)
    cache = {}

    def read(name):
        if name not in cache:
            cache[name] = loadmat(io.BytesIO(archive.read("SourceDataFile/" + name)),
                                  squeeze_me=True, struct_as_record=False)
        return cache[name]

    records = {"train": [], "holdout": []}
    rows = []
    seen = set()
    for split, key in [("train", "train_trajectories"), ("holdout", "holdout_trajectories")]:
        for spec in protocol[key]:
            mat = read(spec["file"])
            if split == "train":
                y = np.asarray(nested(mat, spec["path"]), float)
                t = np.asarray(nested(mat, spec["time_path"]), float)
                light = np.full(t.size, spec["constant_light"], float)
                target = np.full(t.size, np.nan)
            else:
                item = nested(mat, spec["path"])
                y = np.asarray(item.PhotophilicFraction, float)
                t = np.asarray(item.time, float)
                light = np.asarray(item.LightInput, float)
                target = np.asarray(item.Setpoint, float)
            assert y.ndim == t.ndim == light.ndim == 1
            assert y.size == t.size == light.size == target.size
            assert np.all(np.isfinite(y)) and np.all((y > 0) & (y < 1))
            assert t[0] == 0 and np.all(np.diff(t) > 0)
            assert np.all((light >= 0) & (light <= 800))
            fingerprint = hashlib.sha256(np.stack([t, y, light]).tobytes()).hexdigest()
            assert fingerprint not in seen, f"Duplicated trajectory: {spec['id']}"
            seen.add(fingerprint)
            record = dict(spec, t=t, y=y, light=light, target=target,
                          fingerprint=fingerprint, split=split)
            records[split].append(record)
            for i in range(t.size):
                rows.append(dict(trajectory=spec["id"], split=split, time_h=t[i],
                                 photophilic_fraction=y[i], light_digital_au=light[i],
                                 setpoint="" if np.isnan(target[i]) else target[i],
                                 source_file=spec["file"], source_variable=".".join(spec["path"]),
                                 published_figure=spec["published_figure"]))
    dose = nested(read("figure2.mat"), ["figure2", "figure2b"])
    lights = np.asarray(dose.light, float)
    growth = np.asarray([getattr(dose, f"steady_state_growth_rate_replicate{i}")
                         for i in [1, 2, 3]], float)
    write_csv(ROOT / "data" / "measured_trajectories.csv", rows)
    write_csv(ROOT / "data" / "measured_dose_response.csv", [
        dict(replicate=r + 1, light_digital_au=l, growth_rate_per_h=growth[r, j],
             source_file="figure2.mat", source_variable=f"figure2.figure2b.steady_state_growth_rate_replicate{r+1}")
        for r in range(3) for j, l in enumerate(lights)])
    return records, lights, growth


def unpack(p):
    return p[0], p[1], np.exp(p[2]), np.exp(p[3]), p[4]


def step(z, g, target, dt, tau_down, tau_up):
    tau = tau_up if target >= g else tau_down
    a = np.exp(-dt / tau)
    return z + target * dt + (g - target) * tau * (1 - a), target + (g - target) * a


def constant_prediction(p, record):
    dark, bright, tau_down, tau_up, initial = unpack(p)
    target = dark if record["constant_light"] == 0 else bright
    tau = tau_up if target >= initial else tau_down
    t = record["t"]
    z = logit(record["y"][0]) + target * t + (initial - target) * tau * (-np.expm1(-t / tau))
    return expit(z)


def fit_composition(records, protocol):
    lower = np.array([-2, 1e-6, np.log(.05), np.log(.05), -2])
    upper = np.array([-1e-6, 2, np.log(24), np.log(24), 2])

    def residual(p):
        return np.concatenate([(constant_prediction(p, r)[1:] - r["y"][1:]) /
                               np.sqrt(r["y"].size - 1) for r in records])

    attempts = []
    best = None
    for down, up in protocol["models"]["composition_starts_tau_down_up"]:
        initial = np.array([-.15, .15, np.log(down), np.log(up), 0.])
        try:
            result = least_squares(residual, initial, bounds=(lower, upper),
                                   loss="soft_l1", f_scale=.03, max_nfev=3000,
                                   ftol=1e-11, xtol=1e-11, gtol=1e-11)
            attempts.append(dict(start_tau=[down, up], success=bool(result.success),
                                 cost=float(result.cost), nfev=result.nfev,
                                 status=int(result.status), message=result.message,
                                 parameters=unpack(result.x)))
            if best is None or result.cost < best.cost:
                best = result
        except Exception as e:
            attempts.append(dict(start_tau=[down, up], success=False, error=str(e)))
    if best is None:
        raise RuntimeError("All composition fits failed")
    return best, diagnostics(best, lower, upper, attempts)


def hill(light, K, n):
    light = np.asarray(light, float)
    return light ** n / (K ** n + light ** n)


def fit_dose(lights, growth):
    lower = np.array([0, 1e-6, np.log(.1), .1])
    upper = np.array([3, 5, np.log(10000), 5])

    def residual(p):
        pred = p[0] + p[1] * hill(lights, np.exp(p[2]), p[3])
        return (pred[None, :] - growth).ravel()

    attempts = []
    best = None
    for K in [30, 100, 300, 1000]:
        for n in [.5, 1, 2]:
            try:
                r = least_squares(residual, [1, 1.5, np.log(K), n],
                                  bounds=(lower, upper), max_nfev=3000,
                                  ftol=1e-11, xtol=1e-11, gtol=1e-11)
                attempts.append(dict(start_K=K, start_n=n, success=bool(r.success),
                                     cost=float(r.cost), nfev=r.nfev, status=int(r.status),
                                     message=r.message, parameters=[r.x[0], r.x[1], np.exp(r.x[2]), r.x[3]]))
                if best is None or r.cost < best.cost:
                    best = r
            except Exception as e:
                attempts.append(dict(start_K=K, start_n=n, success=False, error=str(e)))
    if best is None:
        raise RuntimeError("All dose fits failed")
    return best, diagnostics(best, lower, upper, attempts)


def diagnostics(result, lower, upper, attempts):
    singular = np.linalg.svd(result.jac, compute_uv=False)
    return dict(success=bool(result.success), cost=float(result.cost), nfev=result.nfev,
                active_mask=result.active_mask.tolist(),
                relative_bound_distance=np.minimum((result.x - lower) / (upper - lower),
                                                   (upper - result.x) / (upper - lower)).tolist(),
                jacobian_singular_values=singular.tolist(),
                jacobian_condition=float(singular[0] / max(singular[-1], 1e-300)),
                parameterization="log-time constants or log-K; condition is scale dependent",
                attempts=attempts)


def predict(p, dose_p, record, model):
    dark, bright, down, up, initial = unpack(p)
    y, t, light = record["y"], record["t"], record["light"]
    if model == "linear":
        response = light / 800
    else:
        K, n = np.exp(dose_p[2]), dose_p[3]
        response = hill(light, K, n) / hill(800, K, n)
    steady = dark + (bright - dark) * response
    free = np.full(y.size, np.nan)
    one = np.full(y.size, np.nan)
    rate = np.full(y.size, np.nan)
    free[0], rate[0] = y[0], initial
    z = logit(y[0])
    for k, dt in enumerate(np.diff(t)):
        old_rate = rate[k]
        z, rate[k+1] = step(z, old_rate, steady[k], dt, down, up)
        free[k+1] = expit(z)
        one[k+1] = expit(step(logit(y[k]), old_rate, steady[k], dt, down, up)[0])
    return free, one, rate


def metric(pred, observed, mask):
    err = pred[mask] - observed[mask]
    if not err.size:
        return dict(n=0, rmse=None, mae=None, max_abs=None)
    return dict(n=int(err.size), rmse=float(np.sqrt(np.mean(err ** 2))),
                mae=float(np.mean(np.abs(err))), max_abs=float(np.max(np.abs(err))))


def self_test():
    errors = []
    for target, initial, down, up, dt in [(.2, -.1, 2., 4., .5), (-.4, .2, 1., 3., 2.)]:
        analytic = np.asarray(step(.3, initial, target, dt, down, up))
        tau = up if target >= initial else down
        numerical = solve_ivp(lambda t, x: [x[1], (target-x[1])/tau],
                              [0, dt], [.3, initial], rtol=1e-12, atol=1e-14).y[:, -1]
        errors.append(float(np.max(np.abs(analytic - numerical))))
    assert max(errors) < 1e-10
    return dict(exact_step_vs_solve_ivp_max_errors=errors, passed=True)


def main():
    started = time.perf_counter()
    protocol_path = ROOT / "protocol.json"
    expected = (ROOT / "PROTOCOL_SHA256.txt").read_text().split()[0]
    assert sha(protocol_path) == expected, "Protocol changed after pre-fit freeze"
    lock = json.loads((ROOT / "LOCK.json").read_text())
    assert sha(__file__) == lock["script_sha256"], "Script changed after lock"
    assert lock["protocol_sha256"] == expected
    assert not (ROOT / "results" / "results.json").exists(), "Do not overwrite completed evaluation"
    protocol = json.loads(protocol_path.read_text())
    test_report = self_test()
    records, lights, growth = read_data(protocol)
    # Fit stage sees training records and training dose response only.
    composition, composition_info = fit_composition(records["train"], protocol)
    dose, dose_info = fit_dose(lights, growth)
    loo_composition = []
    for i, omitted in enumerate(records["train"]):
        result, info = fit_composition([r for j, r in enumerate(records["train"]) if j != i], protocol)
        loo_composition.append(dict(omitted=omitted["id"], x=result.x.tolist(), parameters=unpack(result.x), diagnostics=info))
    loo_dose = []
    for i in range(3):
        result, info = fit_dose(lights, np.delete(growth, i, axis=0))
        loo_dose.append(dict(omitted_replicate=i+1, x=result.x.tolist(), diagnostics=info))
    fit = dict(composition_parameters=dict(zip(["g_dark_per_h", "g_bright_per_h", "tau_down_h", "tau_up_h", "g_initial_per_h"], unpack(composition.x))),
               dose_parameters=dict(offset=dose.x[0], amplitude=dose.x[1], K=np.exp(dose.x[2]), n=dose.x[3]),
               composition_diagnostics=composition_info, dose_diagnostics=dose_info,
               leave_one_trajectory_out=loo_composition, leave_one_dose_replicate_out=loo_dose)
    # Persist training-only result before first heldout scoring.
    write_json(ROOT / "results" / "training_fit.json", fit)
    training_fit_hash = sha(ROOT / "results" / "training_fit.json")
    metrics = []
    predictions = []
    cache = {}
    sensitivity = []
    for split in ["train", "holdout"]:
        for record in records[split]:
            y, t = record["y"], record["t"]
            models = ["linear", "dose_informed"] if split == "holdout" else ["linear"]
            for model in models:
                free, one, rate = predict(composition.x, dose.x, record, model)
                cache[record["id"], model] = (free, one, rate)
                for window, mask in [("full", t > 0), ("through40h", (t > 0) & (t <= 40)), ("after40h", t > 40)]:
                    for mode, pred in [("free_run", free), ("one_step", one)]:
                        base = np.full(y.size, y[0]) if mode == "free_run" else np.r_[np.nan, y[:-1]]
                        m = metric(pred, y, mask)
                        b = metric(base, y, mask)
                        metrics.append(dict(trajectory=record["id"], split=split, model=model,
                                            mode=mode, window=window, **m,
                                            baseline_rmse=b["rmse"], baseline_mae=b["mae"],
                                            terminal_abs_error=float(abs(pred[-1]-y[-1])) if window=="full" else None))
                for k in range(y.size):
                    predictions.append(dict(trajectory=record["id"], split=split, model=model,
                        time_h=t[k], observed_fraction=y[k], light_digital_au=record["light"][k],
                        free_run_fraction=free[k], one_step_fraction="" if k==0 else one[k],
                        predicted_growth_difference_per_h=rate[k]))
                if split == "holdout":
                    candidate_errors = []
                    for item in loo_composition:
                        f, o, _ = predict(np.asarray(item["x"]), dose.x, record, model)
                        candidate_errors.append(dict(kind="omit_training_trajectory", omitted=item["omitted"],
                            free_rmse=metric(f,y,t>0)["rmse"], one_rmse=metric(o,y,t>0)["rmse"]))
                    if model == "dose_informed":
                        for item in loo_dose:
                            f, o, _ = predict(composition.x,np.asarray(item["x"]),record,model)
                            candidate_errors.append(dict(kind="omit_dose_replicate",omitted=item["omitted_replicate"],
                                free_rmse=metric(f,y,t>0)["rmse"],one_rmse=metric(o,y,t>0)["rmse"]))
                    sensitivity.append(dict(trajectory=record["id"],model=model,refits=candidate_errors))
    write_csv(ROOT / "results" / "metrics.csv", metrics)
    write_csv(ROOT / "results" / "predictions.csv", predictions)
    write_json(ROOT / "results" / "sensitivity.json", sensitivity)
    summaries = []
    for model in ["linear", "dose_informed"]:
        for mode in ["free_run", "one_step"]:
            for window in ["full", "through40h", "after40h"]:
                selected = [m for m in metrics if m["split"]=="holdout" and m["model"]==model and m["mode"]==mode and m["window"]==window and m["n"]]
                summaries.append(dict(model=model,mode=mode,window=window,trajectories=len(selected),
                    macro_rmse=float(np.mean([m["rmse"] for m in selected])),
                    macro_mae=float(np.mean([m["mae"] for m in selected])),
                    macro_baseline_rmse=float(np.mean([m["baseline_rmse"] for m in selected])),
                    better_than_baseline_count=sum(m["rmse"]<m["baseline_rmse"] for m in selected),
                    max_trajectory_rmse=max(m["rmse"] for m in selected)))
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.spines.top":False,"axes.spines.right":False})
    fig, axes = plt.subplots(2,4,figsize=(12.4,6.0),sharex=False,sharey=True,layout="constrained")
    for ax,r in zip(axes.flat,records["holdout"]):
        ax.plot(r["t"],r["y"],color="#202020",lw=1.3,label="Measured fraction")
        ax.plot(r["t"],cache[r["id"],"linear"][0],color="#d88920",lw=1,label="Linear-response rollout")
        ax.plot(r["t"],cache[r["id"],"dose_informed"][0],color="#176b96",lw=1,label="Dose-informed rollout")
        ax.plot(r["t"],r["target"],color="#999999",ls=":",lw=.9,label="Published setpoint")
        if r["t"][-1]>40:ax.axvspan(40,r["t"][-1],color="#d6d6d6",alpha=.3)
        ax.set_title(r["id"].replace("closedloop_","").replace("Setpoint"," target").replace("_"," "),fontsize=9)
        ax.set_ylim(-.025,1.025);ax.set_xlabel("Time (h)")
        ax.grid(alpha=.18)
    axes[0,0].set_ylabel("Photophilic cell fraction");axes[1,0].set_ylabel("Photophilic cell fraction")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=4,frameon=False)
    fig.suptitle("Recorded-input prediction on eight heldout E. coli coculture trajectories",fontsize=13)
    fig.savefig(ROOT/"figures"/"heldout_rollouts.pdf")
    fig.savefig(ROOT/"figures"/"heldout_rollouts.png",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8.8,3.6),layout="constrained")
    for r in records["train"]:
        color="#456b99" if r["constant_light"]==0 else "#db8b27"
        axes[0].plot(r["t"],r["y"],".",ms=2,color=color,alpha=.65)
        axes[0].plot(r["t"],constant_prediction(composition.x,r),lw=.8,color=color)
    axes[0].set(xlabel="Time (h)",ylabel="Photophilic cell fraction",title="Six open-loop training trajectories")
    for r in range(3):axes[1].plot(lights,growth[r],"o",ms=3,alpha=.7,label=f"Biological replicate {r+1}")
    grid=np.linspace(0,800,300)
    axes[1].plot(grid,dose.x[0]+dose.x[1]*hill(grid,np.exp(dose.x[2]),dose.x[3]),color="#202020",label="Training-only Hill fit")
    axes[1].set(xlabel="Digital light (arbitrary units)",ylabel="Photophilic growth rate (h$^{-1}$)",title="Independent static-response training")
    axes[1].legend(fontsize=6.8,frameon=False)
    for ax in axes:ax.grid(alpha=.18)
    fig.savefig(ROOT/"figures"/"training_fit.pdf");fig.savefig(ROOT/"figures"/"training_fit.png",dpi=180);plt.close(fig)
    result=dict(protocol_sha256=expected,script_sha256=sha(__file__),source_sha256=protocol["source"]["sha256"],
                training_fit_sha256_before_holdout_evaluation=training_fit_hash,
                train_trajectories=len(records["train"]),holdout_trajectories=len(records["holdout"]),
                training_composition_points=sum(r["y"].size for r in records["train"]),
                holdout_composition_points=sum(r["y"].size for r in records["holdout"]),
                dose_training_observations=int(growth.size),summaries=summaries,
                self_test=test_report,elapsed_seconds=time.perf_counter()-started,
                statement="All predeclared models reported; no selection or refit using heldout targets or outcomes.")
    write_json(ROOT/"results"/"results.json",result)
    write_json(ROOT/"audit"/"source_mapping.json",[{k:r[k] for k in ["id","file","path","published_figure","fingerprint","split"]} for split in ["train","holdout"] for r in records[split]])
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
