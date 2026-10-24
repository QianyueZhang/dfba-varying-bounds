"""List or run the configured simulation and analysis commands for one case."""

from pathlib import Path
import argparse
import json
import os
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    cases = json.loads((ROOT / "cases.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", choices=list(cases))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--step", type=int,
        help="Run only this 1-based command; earlier outputs must already exist.",
    )
    args = parser.parse_args(argv)
    if args.case is None:
        if args.step is not None or args.execute:
            parser.error("Choose a case before using --step or --execute.")
        for key, value in cases.items():
            print(key + ": " + value["description"])
        return

    case = cases[args.case]
    if args.step is not None and not 1 <= args.step <= len(case["commands"]):
        parser.error(f"--step must be between 1 and {len(case['commands'])} for {args.case}.")
    selected = [
        (i, command) for i, command in enumerate(case["commands"], 1)
        if args.step is None or args.step == i
    ]
    for _, command in selected:
        if not (ROOT / command[0]).is_file():
            parser.error(f"Missing case script: {command[0]}")

    env = os.environ.copy()
    for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        env[name] = "1"
    if args.execute:
        for path in case.get("output_directories", []):
            (ROOT / path).mkdir(parents=True, exist_ok=True)
    for i, command in selected:
        print(str(i) + ". python " + shlex.join(command), flush=True)
        if args.execute:
            subprocess.run([sys.executable, *command], cwd=ROOT, env=env, check=True)
    if not args.execute:
        print("Command listing only. Add --execute to run; full studies may take substantial time.")


if __name__ == "__main__":
    main()
