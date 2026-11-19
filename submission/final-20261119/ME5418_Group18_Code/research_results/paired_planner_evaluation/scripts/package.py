"""Make a release ZIP only if every expected raw rollout and replay audit exists."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--out",type=Path,required=True)
out=p.parse_args().out.resolve()
project=Path(__file__).resolve().parents[1]
if not __debug__ or out.exists() or project in out.parents:
    p.error("Use assertions and a new archive path outside the package directory.")
paths={f.name:f for f in (project/"data/episodes").glob("*.json")}
expected={f"{s}-{k}.json" for s in range(5420000,5420010) for k in ("0","1","3","3b32","5","7")}
assert paths.keys()==expected
for s in range(5420000,5420010):
    actual=(project/"audits"/f"replay-{s}.txt").read_text().splitlines()
    required=[f"PASS {k} {json.loads(v.read_text())['steps']} actual transitions" for k,v in sorted(paths.items()) if k.startswith(f"{s}-")]
    assert actual==required,(s,actual)
assert "PASS" in (project/"audits/h1.txt").read_text()
assert "PASS" in (project/"audits/h3.txt").read_text()
with tempfile.TemporaryDirectory() as scratch:
    subprocess.run([sys.executable,str(project/"scripts/tables.py"),"--output",scratch],check=True)
    for f in Path(scratch).glob("*.csv"):
        assert f.read_bytes()==(project/"tables"/f.name).read_bytes(),f.name
subprocess.run([sys.executable,str(project/"scripts/report.py")],check=True)
files=sorted(f for f in project.rglob("*") if f.is_file() and "__pycache__" not in f.parts and not f.name.startswith(".") and f.name!="SHA256SUMS")
manifest=project/"SHA256SUMS"
manifest.write_text("".join(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(project).as_posix()}\n" for f in files),encoding="utf-8")
with zipfile.ZipFile(out,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=8) as z:
    for f in files+[manifest]:
        z.write(f,project.name+"/"+f.relative_to(project).as_posix())
with zipfile.ZipFile(out) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(files)+1
    for f in files+[manifest]:
        assert z.read(project.name+"/"+f.relative_to(project).as_posix())==f.read_bytes()
print(f"PASS: {len(paths)} complete episodes, fully replayed; {len(files)+1} ZIP entries; SHA-256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
