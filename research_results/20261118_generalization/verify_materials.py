"""Verify the byte-for-byte historical files exposed in this browse directory."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
entries = json.loads((root / 'FILE_INDEX.json').read_text())
errors = []
for item in entries:
    path = root / item['uploaded_path']
    if not path.is_file():
        errors.append(f"Missing: {item['uploaded_path']}")
        continue
    data = path.read_bytes()
    if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
        errors.append(f"Mismatch: {item['uploaded_path']}")
if errors:
    raise SystemExit('\n'.join(errors))
print(f'Verified {len(entries)} archived files. This checks file integrity, not model performance.')
