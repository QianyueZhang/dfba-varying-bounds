"""Amount-balanced two-yeast reactor and terminal sensitivities.

This is a new computational extension, not a new biological reconstruction.
The archived paired growth/uptake responses are queried at kappa_i*S, equivalent
to changing Ks_i to Ks_i/kappa_i in the original uptake capacity. No response
is extrapolated. The archived low-S closure uses physical extracellular S.
"""
from pathlib import Path
import json
import time
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp
import kernel

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/flux_table_eps1e-06.json'
FEED_SUBSTRATE = 100.0
VOLUME_CAP = 1.8
DT = 2.0
RATIO_TOL = float(np.log(1.01))
DESIGN_TOL = 0.008


def initial_state(ratio, substrate):
    volume = 0.5
    amounts = volume * 0.002 * np.array([ratio, 1.0]) / (1.0 + ratio)
    return np.r_[volume, np.log(amounts), volume * substrate]


def observable(state):
    v, a, b, q = np.asarray(state)
    return {'volume_L': float(v), 'biomass_amount_gDW': float(np.exp(a)+np.exp(b)),
            'log_ratio': float(a-b), 'ratio': float(np.exp(a-b)),
            'glucose_g_L': float(q/v), 'SC_gDW_L': float(np.exp(a)/v),
            'KP_gDW_L': float(np.exp(b)/v)}


class Reactor:
    def __init__(self, source=SOURCE):
        raw = json.loads(Path(source).read_text())
        self.curves = []
        for species in ('SC', 'KP'):
            rows = raw[species]['rows']
            curve = PchipInterpolator([r['S'] for r in rows],
                                     [[r['growth'], r['glucose_flux']] for r in rows],
                                     axis=0, extrapolate=False)
            self.curves.append((curve, curve.derivative()))
        self.calls = 0
        self.seconds = 0.0
        self.failures = 0

    def rates(self, substrate, kappa):
        physical = max(float(substrate), 0.0)
        factor = min(1.0, physical / 1e-5)
        factor_derivative = 1e5 if 0.0 < physical < 1e-5 else 0.0
        rates, slopes = [], []
        for k, (curve, derivative) in zip(kappa, self.curves):
            effective = k * physical
            if effective > 100.0 + 1e-10 or effective < 0.0:
                raise ValueError('Effective substrate outside archived table domain')
            base = curve(effective)
            rates.append(base * factor)
            slopes.append(derivative(effective)*k*factor + base*factor_derivative)
        return np.asarray(rates), np.asarray(slopes)

    def predict(self, state, controls, kappa=(1.0, 1.0), *, gradients=False,
                tight=False, duration=DT, record=False):
        started = time.perf_counter()
        self.calls += 1
        controls = np.atleast_1d(controls).astype(float)
        n = len(controls)
        y = np.asarray(state, float).copy()
        sens = np.zeros((4, n))
        trajectory = [y.tolist()]
        times = [0.0]
        minimum_s = float(y[3]/y[0])
        maximum_s = minimum_s
        ramp_calls = 0
        rhs_calls = 0
        try:
            for j, feed in enumerate(controls):
                feed_tangent=np.eye(n)[j]
                kappa_array=np.asarray(kappa,dtype=float)
                def rhs(t, extended):
                    nonlocal ramp_calls, rhs_calls
                    rhs_calls += 1
                    if getattr(self,'fast',False):
                        ramp_calls += int(extended[3]/extended[0]<1e-5)
                        return kernel.rhs(extended,feed,kappa_array,feed_tangent,gradients,*self.kernel_args)
                    yy = extended[:4]
                    v, l1, l2, q = yy
                    s = q/v
                    r, dr = self.rates(s, kappa)
                    ramp_calls += int(s < 1e-5)
                    m = np.exp([l1, l2])
                    f = np.array([feed, r[0, 0], r[1, 0],
                                  feed*FEED_SUBSTRATE + 0.18*np.dot(r[:, 1], m)])
                    if not gradients:
                        return f
                    jac = np.zeros((4, 4))
                    jac[1:3, 0] = -dr[:, 0]*s/v
                    jac[1:3, 3] = dr[:, 0]/v
                    uptake_slope = 0.18*np.dot(dr[:, 1], m)
                    jac[3, 0] = -uptake_slope*s/v
                    jac[3, 1:3] = 0.18*r[:, 1]*m
                    jac[3, 3] = uptake_slope/v
                    ds = jac @ extended[4:].reshape(4, n)
                    ds[0, j] += 1.0
                    ds[3, j] += FEED_SUBSTRATE
                    return np.r_[f, ds.ravel()]

                y0 = np.r_[y, sens.ravel()] if gradients else y
                sol = solve_ivp(rhs, (j*duration, (j+1)*duration), y0,
                                method='DOP853' if tight else 'LSODA', rtol=5e-12 if tight else 2e-9,
                                atol=5e-14 if tight else 2e-11,
                                max_step=0.05 if tight else 0.15)
                if not sol.success:
                    raise RuntimeError(sol.message)
                if np.min(sol.y[3]/sol.y[0]) < -2e-7:
                    raise RuntimeError('Negative physical substrate concentration')
                minimum_s = min(minimum_s, float(np.min(sol.y[3]/sol.y[0])))
                maximum_s = max(maximum_s, float(np.max(sol.y[3]/sol.y[0])))
                y = sol.y[:4, -1]
                if gradients:
                    sens = sol.y[4:, -1].reshape(4, n)
                if record:
                    trajectory.extend(sol.y[:4, 1:].T.tolist())
                    times.extend(sol.t[1:].tolist())
            amounts = np.exp(y[1:3])
            total = float(np.sum(amounts))
            result = {'state': y, 'z': float(y[1]-y[2]), 'logB': float(np.log(total)),
                      'B': total, 'V': float(y[0]), 'min_S': minimum_s,
                      'max_S': maximum_s, 'ramp_rhs_calls': ramp_calls,
                      'rhs_calls': rhs_calls}
            if gradients:
                result.update(jz=sens[1]-sens[2],
                              jlogB=(amounts @ sens[1:3])/total,
                              jV=sens[0], sensitivities=sens)
            if record:
                result.update(trajectory=trajectory, times=times)
            return result
        except Exception:
            self.failures += 1
            raise
        finally:
            self.seconds += time.perf_counter()-started
