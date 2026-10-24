"""Freeze fresh continuous reproduction inputs without altering archived records.

Run only in a fresh working directory after preflight and six grid41 outputs:
python scripts/prepare_continuous_reproduction.py --reference path/to/archived.json
"""
from pathlib import Path
import argparse,datetime,hashlib,json
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args();destination=ROOT/'protocols/continuous_protocol.json'
    note_path=ROOT/'protocols/continuous_reporting_note.json'
    if destination.exists() or note_path.exists():
        raise SystemExit('Refusing to overwrite an active continuous protocol or reporting note; use a fresh reproduction directory.')
    source=args.reference.resolve();protocol=json.loads(source.read_text())
    gate=json.loads((ROOT/'audit/grid_observation_integrator_gate.json').read_text())
    assert gate['complete'] and gate['passed'],'Independent numerical gate must pass before outcomes.'
    for name,digest in gate['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in protocol['source_sha256'].items():
        assert sha(ROOT/name)==digest,'Archived numerical source changed; declare a new study instead of an exact-source reproduction: '+name
    for name in ['data/prescribed_roots.json','data/flux_table_eps1e-06.json']:
        assert sha(ROOT/name)==protocol['input_sha256'][name],'Scientific input changed: '+name
    for kind in ['source_sha256','input_sha256']:
        protocol[kind]={name:sha(ROOT/name) for name in protocol[kind]}
    protocol['reproduction_of_protocol_sha256']=sha(source)
    protocol['reproduction_note']='Same declared parameters; fresh input/source hashes. Timing metadata and numerical results need not be byte-identical across runs or environments.'
    protocol['frozen_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    protocol['frozen_before_evaluation']=True
    assert not any((ROOT/'results/continuous').glob('*_target_*.json')),'Fresh outcome directory required.'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(protocol,indent=2)+'\n')
    note=dict(recorded_utc=protocol['frozen_utc'],status='Reporting gate declared before this reproduction executes',
              formal_success='All nine paths complete AND selected.valid AND all_physical AND error threshold satisfied.',
              applies_to='All72 assigned method-problems',
              original_protocol_sha256=sha(destination),
              sources={'scripts/continuous_report.py':sha(ROOT/'scripts/continuous_report.py')},
              optimizer_status_reporting='DE budget exhaustion is distinct from other nonconvergence, SLSQP or scalar nonconvergence, and exceptions.',
              comparison_scale='2e-7 is a descriptive paired comparison scale, not a statistical test or changed success threshold.')
    note_path.write_text(json.dumps(note,indent=2)+'\n')
    print('Fresh continuous protocol:',destination)
    print('sha256:',sha(destination))

if __name__=='__main__':main()
