"""Build-time only: fetch an immutable public model and verify every artifact."""
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

manifest_path = Path(__file__).with_name("guardrails-model.json")
manifest = json.loads(manifest_path.read_text())
target = Path("/opt/guardrails-embeddings")
target.mkdir(parents=True, exist_ok=True)
for name, expected in manifest["files"].items():
    url = f'https://huggingface.co/{manifest["repository"]}/resolve/{manifest["revision"]}/{name}'
    with urllib.request.urlopen(url, timeout=120) as response, (target / name).open("wb") as output:
        shutil.copyfileobj(response, output)
    with (target / name).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, name
shutil.copyfile(manifest_path, target / "manifest.json")
