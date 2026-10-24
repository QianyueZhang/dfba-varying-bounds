"""Create a new source-only study directory without executing the campaign.

Original declarations are kept under archived_protocols. The ordered command
manifest makes new declarations against regenerated inputs before evaluation.
The destination must not exist; this command never resets an old run.
"""
from pathlib import Path
import argparse, ast, datetime, hashlib, json, shutil, sys

ROOT=Path(__file__).resolve().parents[1]
VERSIONS={'numpy':'2.2.2','scipy':'1.15.1','numba':'0.68.0','llvmlite':'0.50.0','matplotlib':'3.8.4'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def steps():
    rows=[]
    def add(name,argv,purpose):rows.append(dict(order=len(rows)+1,id=name,argv=argv,purpose=purpose))
    add('dependencies',['python','-m','pip','install','-r','requirements-reproduction.txt'],
        'Install the recorded scientific runtime in a Python3.11 environment; environment creation is user-selected.')
    add('preflight_declare',['python','audit/grid_observation_preflight.py','--declare'],'Declare fresh integrator check before any flow outcome.')
    add('preflight_execute',['python','audit/grid_observation_preflight.py','--execute'],'Run independent amount checks; require complete=true and passed=true before continuing.')
    add('grid_declare',['python','scripts/grid_observation.py','--freeze'],'Declare the original41/81/161 design using this directory’s source hashes.')
    add('grid_amend',['python','scripts/grid_observation.py','--amend'],'Declare adaptive321 plus60min budget and sensing-frontier analysis before tree evaluation; the v1 snapshot is retained historical source evidence.')
    add('audit_declare',['python','scripts/prepare_audit_reproduction.py','--reference-dir','archived_protocols'],'Clone the archived audit selection rules, refresh source/gate/grid-protocol hashes, and explicitly record a reproduction declaration.')
    add('algebra_check',['python','audit/grid_observation_algebra_review.py'],'The script writes a fresh synthetic-test declaration then exhaustively checks finite-game algebra.')
    add('grid_campaign',['python','scripts/grid_observation.py','--run'],'Run all primary grids and every triggered adaptive321 root; preserve budget-limited incomplete roots.')
    add('primary_audit_bind',['python','audit/grid_observation_deploy.py','--bind'],'Freeze the regenerated primary results, exact case list, policies and source hashes before independent deployment.')
    add('primary_audit_execute',['python','audit/grid_observation_deploy.py','--execute'],'Independently execute all assigned primary cases; preserve strict endpoint label differences and failed audit status.')
    add('adaptive_audit_bind',['python','audit/grid_observation_deploy.py','--adaptive','--bind'],'Bind every completed adaptive root and explicitly retain any unresolved root.')
    add('adaptive_audit_execute',['python','audit/grid_observation_deploy.py','--adaptive','--execute'],'Independently execute the adaptive cases selected by the unchanged rule.')
    add('set_distance_declare',['python','scripts/diagnostic_set_hausdorff.py','--freeze'],'Retain the explicitly post hoc status of the set-distance diagnostic; it does not change the original gates.')
    add('set_distance_execute',['python','scripts/diagnostic_set_hausdorff.py','--run'],'Recount geometric distances from saved computed sets; no new flow integration.')
    add('continuous_declare',['python','scripts/prepare_continuous_reproduction.py','--reference','archived_protocols/continuous_protocol.json'],'Clone the continuous design and bind new gate/grid41 inputs, including their fresh timing metadata, before optimization.')
    add('continuous_campaign',['python','scripts/continuous_run.py','--workers','3'],'Run both continuous methods on all36 fixed problems and retain all candidate/status/deployment records.')
    add('continuous_report',['python','scripts/continuous_report.py'],'Apply physical-validity and completion gates to every continuous success count.')
    add('continuous_figure',['python','scripts/continuous_figure.py'],'Plot independent continuous outcomes for all36 fixed problems.')
    add('grid_report',['python','scripts/report_grid_observation.py'],'Recount all completed grid/observation/audit results and generate scientific figures.')
    return rows


def prepare(destination,source=ROOT):
    source=Path(source).resolve();destination=Path(destination).expanduser().absolute()
    if destination.exists() or destination.is_symlink():raise FileExistsError('Destination already exists; choose a new directory')
    required=['scripts/prepare_audit_reproduction.py','scripts/prepare_continuous_reproduction.py',
              'scripts/continuous_figure.py','protocols/frozen_grid_observation_v1.py',
              'protocols/grid_observation.json','protocols/grid_observation_amendment.json',
              'protocols/continuous_protocol.json','protocols/audit_grid_observation.json']
    for rel in required:
        if not (source/rel).is_file():raise FileNotFoundError('Required source/declaration missing: '+rel)
    if not destination.parent.exists():raise FileNotFoundError('Destination parent must already exist')
    destination.mkdir()
    for name in ['scripts','audit','vendor','data','protocols','archived_protocols','results','figures','manuscript']:
        (destination/name).mkdir()
    copied={}
    for folder in ['scripts','audit','vendor']:
        for p in sorted((source/folder).glob('*.py')):
            target=destination/folder/p.name;shutil.copyfile(p,target);copied[str(target.relative_to(destination))]=sha(p)
    for p in sorted((source/'data').glob('*.json')):
        target=destination/'data'/p.name;shutil.copyfile(p,target);copied[str(target.relative_to(destination))]=sha(p)
    for p in sorted((source/'protocols').iterdir()):
        if p.is_file() and p.suffix in ['.json','.py']:
            target=destination/'archived_protocols'/p.name;shutil.copyfile(p,target);copied[str(target.relative_to(destination))]=sha(p)
    historical=source/'protocols/frozen_grid_observation_v1.py'
    shutil.copyfile(historical,destination/'protocols'/historical.name)
    copied['protocols/'+historical.name]=sha(historical)
    requirements='\n'.join(f'{name}=={version}' for name,version in VERSIONS.items())+'\n'
    (destination/'requirements-reproduction.txt').write_text(requirements)
    manifest=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source_files_sha256=copied,
                  copied_scientific_results=False,copied_audit_outcomes=False,
                  historical_protocols='archived_protocols; unchanged copies, never represented as fresh declarations',
                  active_protocols='Fresh declarations generated by RUN_COMMANDS.json; the historicalv1source snapshot is retained unchanged',
                  exact_output_equality='Not promised: solver/platform floating point, timestamps, timings, candidate ordering and strict boundary classifications can differ. New input hashes must be frozen before each new evaluation.',
                  python='3.11',recorded_dependencies=VERSIONS)
    write(destination/'REPRODUCTION_MANIFEST.json',manifest)
    write(destination/'RUN_COMMANDS.json',dict(working_directory='.',
        environment={'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','NUMBA_NUM_THREADS':'1'},
        command_interpreter='Each argv is a direct process argument list, not shell code. Use the selected environment’s python executable.',
        order=steps(),
        required_gate='The numerical preflight must pass before tree/continuous evaluation.',
        audit_semantics='A completed deployment audit can legitimately have passed=false because strict boundary labels differ. Preserve that result; do not discard cases or rerun until it passes.',
        overwrite_policy='Never overwrite an existing declaration/outcome. For a new campaign, prepare another new directory.',
        runtime_note='This deliberately sequential recipe prioritizes reproducibility. Scientific outputs are not expected to reproduce original overlapping-job wall times.'))
    (destination/'REPRODUCE.md').write_text('''# Source-only reproduction

This directory contains source, scientific inputs and archived declarations. It contains no scientific result JSON, transition cache, candidate ledger, or audit outcome from the original campaign.

Use a Python3.11 environment. Run the argument lists in `RUN_COMMANDS.json` in their numbered order, with this directory as the working directory and the listed thread limits. The first step installs the recorded package versions. No machine-specific Python path or executable path is embedded in the commands.

The original declarations remain unchanged under `archived_protocols`. Fresh declarations bind this directory’s regenerated inputs before evaluation. In particular, the continuous input hashes include the new numerical gate and grid41 metadata, including timings; copying the old active continuous protocol would therefore reject a legitimate fresh run. The preparation helper creates a new declaration with the same design and new hashes.

The adaptive321 stage retains its original60minute additional wall-clock budget. A slower machine can therefore produce more unresolved roots; those are results to retain, not permission to silently change the budget. Timing and exact floating-point boundary labels are not promised to match the original archive byte for byte.

Independent deployment must retain all failed checks. A completed audit with strict endpoint label differences remains `passed=false`; successful numerical state comparisons do not erase that status. Fixed cases, selected component points and disturbance/measurement paths have separate denominators.

`python scripts/prepare_reproduction.py --validate .` checks this source-only preparation without running simulation or optimization. It is intended before executing the campaign, because a completed run will contain outcomes and thus cease to be source-only.
''')
    return validate(destination)


def validate(destination):
    destination=Path(destination).resolve();manifest=json.loads((destination/'REPRODUCTION_MANIFEST.json').read_text())
    for rel,expected in manifest['source_files_sha256'].items():
        if sha(destination/rel)!=expected:raise RuntimeError('Copied source changed: '+rel)
    for p in destination.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
    if any((destination/'results').iterdir()):raise RuntimeError('Scientific results were copied or generated')
    if list((destination/'audit').glob('*.json')):raise RuntimeError('Audit outcomes/declarations were copied')
    if list((destination/'protocols').glob('*.json')):raise RuntimeError('Old active protocol JSON was copied')
    commands=json.loads((destination/'RUN_COMMANDS.json').read_text())
    for step in commands['order']:
        argv=step['argv']
        if len(argv)>1 and argv[1].endswith('.py') and not (destination/argv[1]).is_file():raise FileNotFoundError(argv[1])
    return dict(valid=True,destination=str(destination),copied_source_files=len(manifest['source_files_sha256']),
                commands=len(commands['order']),scientific_outcomes_copied=0,audit_outcomes_copied=0,
                validation='SHA256 source equality, in-memory Python syntax parsing, command-path checks, and absence of active declarations/outcomes; no simulations or optimizations run')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--destination');group.add_argument('--validate')
    parser.add_argument('--source-root',default=str(ROOT))
    args=parser.parse_args()
    result=prepare(args.destination,args.source_root) if args.destination else validate(args.validate)
    print(json.dumps(result,indent=2))
