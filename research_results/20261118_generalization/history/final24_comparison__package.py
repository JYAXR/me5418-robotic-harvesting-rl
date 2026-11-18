"""Self-contained evaluation bundle with frozen models and all outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=HERE/'results/run01';audit=json.loads((out/'audit.json').read_text())
    assert audit['passed'] and audit['episodes']==3192
    assert (HERE/'SIMPLE_RESULTS.md').is_file() and (HERE/'RESULTS.md').is_file()
    protocol=json.loads((out/'protocol.json').read_text())
    paths={ROOT/p for p in protocol['hashes']}
    paths|={p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    paths|={ROOT/p for p in ('README.md','pyproject.toml','environment.yml','tests/test_scene_cache.py')}
    assert all(p.is_file() for p in paths)
    for path,digest in protocol['hashes'].items():assert sha(ROOT/path)==digest
    manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
    manifest['START_HERE.md']=sha(HERE/'DOWNLOAD.md')
    with zipfile.ZipFile(args.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(paths):z.write(p,p.relative_to(ROOT))
        z.write(HERE/'DOWNLOAD.md','START_HERE.md')
        z.writestr('COMPARISON_MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(args.output) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    print(json.dumps(dict(path=str(args.output),bytes=args.output.stat().st_size,files=len(manifest),sha256=sha(args.output),verified=True),indent=2))


if __name__=='__main__': main()
