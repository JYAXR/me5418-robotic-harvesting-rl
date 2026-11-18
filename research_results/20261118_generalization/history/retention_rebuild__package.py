"""Package the audited rerun without copying unrelated experiments or environments."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
RESULT = HERE / 'results/run01'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['models', 'full'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert json.loads((RESULT / 'audit.json').read_text())['passed']
    lock = json.loads((RESULT / 'selection_lock.json').read_text())
    common = ['README.md', 'pyproject.toml', 'environment.yml',
              'configs/environment_multiroute.yaml',
              'experiments/fresh_coordinate_dqn/policy.py',
              'experiments/fresh_coordinate_dqn/run.py',
              'experiments/fresh_coordinate_dqn/evaluate.py',
              'experiments/online_pilot/run.py']
    paths = {ROOT / p for p in common} | set((ROOT / 'src').rglob('*.py'))
    paths |= {HERE / p for p in ['README.md', 'RESULTS.md', 'TESTING.md', 'DOWNLOAD.md']}
    paths |= {RESULT / p for p in ['summary.json', 'selection_lock.json', 'audit.json',
                                  'historical_hash_comparison.json', 'protocol.json']}
    paths |= set((RESULT / 'evaluation').rglob('*.json'))
    for item in lock['models'].values():
        path = ROOT / item['path']
        assert sha(path) == item['sha256']
        paths.add(path)
    if args.kind == 'full':
        paths |= {p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        paths |= {ROOT / p for p in [
            'experiments/fresh_penalty_coordinates/run.py',
            'experiments/fresh_coordinate_dqn/results/run01/relative/initial_state.pt',
            'docs/experiments/029_training_coverage/evidence/expanded_demonstrations.npz',
            'docs/experiments/029_training_coverage/evidence/dataset.json',
            'docs/experiments/027_step6_supervision/evidence/recovery_sample.json',
            'tests/test_scene_cache.py']}
    assert all(p.is_file() for p in paths)
    manifest = dict(kind=args.kind, files={str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)},
                    scope='Isolated rerun; original source bundled unchanged. Same historical test seeds, not a new blind result.',
                    training_states_included=args.kind == 'full')
    manifest['files']['START_HERE.md'] = sha(HERE / 'DOWNLOAD.md')
    # Exclusive creation avoids overwriting a package the user may already have.
    with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for p in sorted(paths):
            archive.write(p, p.relative_to(ROOT))
        archive.write(HERE / 'DOWNLOAD.md', 'START_HERE.md')
        archive.writestr('REBUILD_MANIFEST.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        for name, digest in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    print(json.dumps(dict(path=str(args.output), bytes=args.output.stat().st_size,
                          files=len(manifest['files']), sha256=sha(args.output), verified=True), indent=2))


if __name__ == '__main__':
    main()
