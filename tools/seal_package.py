"""Maintainer-only build step. Never run this to bypass an integrity error."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = []
    for name in ('models', 'datasets', 'diffusion', 'scripts', 'reference', 'configs', 'tests', 'tools'):
        files.extend(p for p in (ROOT / name).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.extend(p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py', '.md', '.txt', '.sh', '.ps1'))
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    payload = {'format': 1, 'files': hashes,
               'scope': 'Code and reference metadata only. No dataset, weights, or reported results.'}
    (ROOT / 'package_manifest.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print('Sealed %d static files' % len(hashes))


if __name__ == '__main__':
    main()
