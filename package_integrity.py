"""Verify the sealed portable source tree; never rewrite the seal at runtime."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_package():
    manifest = json.loads((ROOT / 'package_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format') != 1 or not manifest.get('files'):
        raise ValueError('Missing/invalid portable package manifest')
    for relative, expected in manifest['files'].items():
        path = ROOT / relative
        if path.is_symlink() or ROOT not in path.resolve().parents or not path.is_file():
            raise ValueError('Invalid package source: ' + relative)
        if file_hash(path) != expected:
            raise ValueError('Package source changed: ' + relative)
    return manifest
