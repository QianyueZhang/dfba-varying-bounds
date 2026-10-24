#!/usr/bin/env python3
"""Run the constrained biomass/composition simulation and surrogate study."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
REFERENCE = PACKAGE / 'reference'
SCRIPTS = PACKAGE / 'scripts'


def environment():
    env = os.environ.copy()
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = '1'
    env['MPLBACKEND'] = 'Agg'
    env['MPLCONFIGDIR'] = str(REPO / 'local_verification/matplotlib_cache')
    return env


def write_new(path, value):
    path = path.resolve()
    if path.is_relative_to(REFERENCE) or path.exists():
        raise ValueError('Choose a new output file outside the supplied reference records.')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    verify = sub.add_parser('verify', help='Check supplied data, policy arithmetic and fitted JSON predictors')
    verify.add_argument('--integrate', action='store_true', help='Also integrate all 180 supplied policies at tighter tolerances')
    verify.add_argument('--output', type=Path)
    simulate = sub.add_parser('simulate', help='Integrate a specified seven-interval feed policy')
    simulate.add_argument('--feeds', type=float, nargs=7, required=True, metavar='L_PER_H')
    simulate.add_argument('--ratio', type=float, default=1.0)
    simulate.add_argument('--glucose', type=float, default=20.0, metavar='G_PER_L')
    simulate.add_argument('--target', type=float, default=1.0)
    simulate.add_argument('--output', type=Path)
    study = sub.add_parser('study', help='Train/select predictors and execute the full 180-policy campaign')
    study.add_argument('--output', type=Path, required=True, help='New directory; the completed reference record is never overwritten')
    figures = sub.add_parser('figures', help='Generate figures and numerical tables from the completed campaign')
    figures.add_argument('--output', type=Path, default=REPO/'local_verification/control_figures')
    figures.add_argument('--study', type=Path, default=REFERENCE, help='Completed study directory')
    balances = sub.add_parser('balance-check', help='Independent amount balances for all 180 policies and tolerance refinement')
    balances.add_argument('--output', type=Path, default=REPO/'local_verification/control_balances')
    direct = sub.add_parser('direct', help='Direct KKT check; requires the four locally prepared yeast model inputs')
    direct.add_argument('--input', type=Path)
    direct.add_argument('--check-point', type=float)
    direct.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    env = environment()
    if args.command == 'verify':
        cmd = [sys.executable, str(SCRIPTS/'verify.py')]
        if args.integrate:
            cmd += ['--integrate']
        if args.output:
            cmd += ['--output', str(args.output.resolve())]
        subprocess.run(cmd, check=True, env=env)
    elif args.command == 'simulate':
        for key, value in env.items():
            if key.endswith('NUM_THREADS') or key == 'VECLIB_MAXIMUM_THREADS':
                os.environ[key] = value
        import math
        if not all(math.isfinite(x) for x in [*args.feeds, args.ratio, args.glucose, args.target]):
            p.error('All inputs must be finite.')
        if any(u < 0 or u > .2 for u in args.feeds) or sum(args.feeds) > .65 + 1e-12:
            p.error('Each feed must be in [0,0.2] L/h and their sum must be at most 0.65 L/h (Vf <= 1.8 L).')
        if not .5 <= args.ratio <= 2 or not 15 <= args.glucose <= 25 or args.target not in [.5, 1., 2.]:
            p.error('Study domain: initial ratio 0.5–2, glucose 15–25 g/L, target 0.5, 1 or 2.')
        sys.path.insert(0, str(SCRIPTS))
        from simulator import Reactor, initial_from_condition
        q = Reactor().simulate(args.feeds, initial_from_condition([args.ratio, args.glucose]), rtol=1e-9, atol=1e-11)
        q['controls_L_h'] = args.feeds
        q['initial_ratio'] = args.ratio
        q['initial_glucose_g_L'] = args.glucose
        q['target_ratio'] = args.target
        q['terminal_names'] = ['V_L', 'X_SC_gDW_L', 'X_KP_gDW_L', 'S_g_L']
        q['feasible'] = abs(q['z']-math.log(args.target)) <= math.log(1.01) and q['terminal'][0] <= 1.8+1e-8
        if args.output:
            write_new(args.output, q)
        print(json.dumps(q, indent=2))
    elif args.command == 'study':
        output = args.output.resolve()
        if output.is_relative_to(PACKAGE) or output.exists():
            p.error('Choose a new study directory outside the distributed experiment.')
        output.mkdir(parents=True)
        (output/'data').mkdir()
        (output/'results').mkdir()
        for source in REFERENCE.glob('protocol*.json'):
            (output/source.name).write_bytes(source.read_bytes())
        env['DFBA_CONTROL_OUTPUT'] = str(output)
        for name in ('run_study.py', 'summarize_results.py'):
            subprocess.run([sys.executable, str(SCRIPTS/name)], env=env, check=True)
        print('Study saved in ' + str(output))
    elif args.command == 'figures':
        if args.output.resolve().is_relative_to(REFERENCE):
            p.error('Choose a figure directory outside the supplied reference record.')
        env['DFBA_CONTROL_OUTPUT'] = str(args.study.resolve())
        env['DFBA_CONTROL_FIGURES'] = str(args.output.resolve())
        subprocess.run([sys.executable, str(SCRIPTS/'render_figures.py')], env=env, check=True)
    elif args.command == 'balance-check':
        if args.output.resolve().is_relative_to(REFERENCE):
            p.error('Choose an audit directory outside the supplied reference records.')
        env['DFBA_CONTROL_AUDIT'] = str(args.output.resolve())
        for name in ('check_tolerance_convergence.py', 'audit_balances.py'):
            subprocess.run([sys.executable, str(SCRIPTS/name)], env=env, check=True)
    elif args.command == 'direct':
        if (args.input is None) == (args.check_point is None):
            p.error('Choose exactly one of --input or --check-point.')
        output = args.output.resolve()
        if output.is_relative_to(REFERENCE) or output.exists() or output.with_name(output.stem+'_trajectory.csv').exists():
            p.error('Choose a new output outside the supplied reference records.')
        cmd = [sys.executable, str(SCRIPTS/'direct_verify.py'), '--root', str(REPO), '--output', str(output)]
        cmd += ['--input', str(args.input.resolve())] if args.input else ['--check-point', str(args.check_point)]
        subprocess.run(cmd, env=env, check=True)


if __name__ == '__main__':
    main()
