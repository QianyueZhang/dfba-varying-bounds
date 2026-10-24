"""Declare an unchanged audit design for a fresh source-only reproduction.

Run after the fresh numerical preflight and grid --freeze/--amend, and before
grid --run. Original dated declarations belong in archived_protocols. This
helper does not run a controller, inspect controller outcomes, or copy results.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import json


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def prepare(reference_dir, check_only=False):
    reference_dir = Path(reference_dir)
    if not reference_dir.is_absolute():
        reference_dir = ROOT / reference_dir
    design_ref = reference_dir / 'audit_grid_observation.json'
    amendment_ref = reference_dir / 'audit_grid_observation_amendment.json'
    design_out = ROOT / 'protocols/audit_grid_observation.json'
    amendment_out = ROOT / 'protocols/audit_grid_observation_amendment.json'
    for path in [design_out, amendment_out]:
        if path.exists():
            raise FileExistsError(f'Active declaration already exists: {path}')
    existing_outcomes = list((ROOT / 'results').glob('grid_observation*.json'))
    existing_outcomes += list((ROOT / 'results/trees').glob('*'))
    existing_outcomes += [p for p in (ROOT / 'audit').glob('grid_observation*.json')
                          if p.name != 'grid_observation_integrator_gate.json']
    if existing_outcomes:
        raise FileExistsError('Declare before controller/algebra outcomes or audit execution records: '
                              + ', '.join(str(p) for p in existing_outcomes))

    design = read(design_ref)
    amendment = read(amendment_ref)
    if amendment['original_audit_protocol_sha256'] != sha(design_ref):
        raise RuntimeError('Archived amendment does not bind archived audit design')
    expected = {**design['source_sha256'], **amendment['source_sha256_overrides']}
    # Protocols and the gate are freshly declared/generated. Scientific source,
    # prescribed data and the historical v1 snapshot must remain byte-identical.
    generated = {'protocols/grid_observation.json',
                 'protocols/grid_observation_amendment.json',
                 'audit/grid_observation_integrator_gate.json'}
    for name, digest in expected.items():
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if name not in generated and sha(path) != digest:
            raise RuntimeError('Archived scientific source differs: ' + name)

    gate = read(ROOT / 'audit/grid_observation_integrator_gate.json')
    if not gate.get('complete') or not gate.get('passed') or not gate.get('inputs_unchanged'):
        raise RuntimeError('Fresh numerical preflight must be complete, passed and unchanged')
    if gate['protocol_sha256'] != sha(ROOT / 'protocols/audit_numerical_preflight.json'):
        raise RuntimeError('Numerical gate does not bind the active preflight declaration')
    for name, digest in gate['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('Fresh preflight source lock mismatch: ' + name)
    fresh_main = read(ROOT / 'protocols/grid_observation_amendment.json')
    if fresh_main['original_protocol_sha256'] != sha(ROOT / 'protocols/grid_observation.json'):
        raise RuntimeError('Fresh grid amendment does not bind fresh grid declaration')
    for name, digest in fresh_main['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('Fresh grid source lock mismatch: ' + name)

    now = datetime.now(timezone.utc).isoformat()
    provenance = {
        'purpose': 'Fresh reproduction declaration; original scientific selection and thresholds unchanged.',
        'archived_design_sha256': sha(design_ref),
        'archived_amendment_sha256': sha(amendment_ref),
        'helper_sha256': sha(Path(__file__)),
        'outcome_policy': 'No original controller or audit outcomes copied; fresh numerical gate required.',
    }
    active_design = copy.deepcopy(design)
    active_design['declared_utc'] = now
    active_design['status'] = 'Fresh reproduction declaration before any controller outcome evaluation'
    active_design['source_sha256'] = {name: sha(ROOT / name) for name in design['source_sha256']}
    active_design['reproduction'] = provenance
    active_amendment = copy.deepcopy(amendment)
    active_amendment['declared_utc'] = now
    active_amendment['reason'] = ('Fresh reproduction of the archived adaptive audit amendment; '
                                'selection, thresholds, failure retention and scientific sources unchanged.')
    active_amendment['main_amendment_sha256'] = sha(ROOT / 'protocols/grid_observation_amendment.json')
    active_amendment['source_sha256_overrides'] = {
        name: sha(ROOT / name) for name in amendment['source_sha256_overrides']}
    active_amendment['reproduction'] = provenance
    design_text = json.dumps(active_design, indent=2, allow_nan=False) + '\n'
    active_amendment['original_audit_protocol_sha256'] = hashlib.sha256(design_text.encode()).hexdigest()
    amendment_text = json.dumps(active_amendment, indent=2, allow_nan=False) + '\n'
    if not check_only:
        with design_out.open('x') as stream:
            stream.write(design_text)
        with amendment_out.open('x') as stream:
            stream.write(amendment_text)
    print(json.dumps({'checked': True, 'written': not check_only,
                      'design': str(design_out), 'amendment': str(amendment_out),
                      'scientific_rules_unchanged': True}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', default='archived_protocols')
    parser.add_argument('--check', action='store_true', help='Validate inputs without writing declarations')
    args = parser.parse_args()
    prepare(args.reference_dir, args.check)
