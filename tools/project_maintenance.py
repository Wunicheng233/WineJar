#!/usr/bin/env python3
"""Offline inventory, documentation checks, and narrowly scoped cache cleanup.

Never imports robot/camera modules. No changes unless clean --apply is supplied.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', 'mujoco_xarm6', '.venv', 'venv', 'models', 'node_modules'}
CACHES = {'__pycache__', '.pytest_cache', '.ruff_cache', '.mypy_cache'}


def walk(root=ROOT):
    for parent, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED and not (Path(parent)/d).is_symlink())
        yield Path(parent), dirs, sorted(files)


def cache_paths():
    result = []
    for parent, dirs, files in walk():
        for name in list(dirs):
            if name in CACHES:
                result.append(parent/name)
                dirs.remove(name)
        for name in files:
            if name.endswith(('.pyc', '.pyo')) and not (parent/name).is_symlink():
                result.append(parent/name)
    return result


def file_size(path):
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink())


def broken_links():
    broken = []
    # Markdown links only; fenced sample commands and historical literal paths
    # are not filesystem dependencies. Anchors are left to the renderer.
    for parent, _dirs, files in walk():
        if ROOT/'xarm_sdk_control/data' in parent.parents or parent == ROOT/'xarm_sdk_control/data':
            continue
        for name in files:
            if not name.endswith('.md'):
                continue
            p = parent/name
            content = re.sub(r'```.*?```', '', p.read_text(encoding='utf-8'), flags=re.S)
            for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)', content):
                link = link.strip('<>').split('#', 1)[0]
                if not link or re.match(r'^[a-zA-Z]+:', link):
                    continue
                target = Path(unquote(link)) if link.startswith('/') else parent/unquote(link)
                if not target.exists():
                    broken.append(dict(document=p.relative_to(ROOT).as_posix(), target=link))
    return broken


def evidence_checks():
    problems = []
    folder = ROOT/'docs/evidence/2026-10-14'
    for entry in json.loads((folder/'index.json').read_text())['files']:
        snapshot = folder/entry['snapshot']
        if not snapshot.exists() or hashlib.sha256(snapshot.read_bytes()).hexdigest() != entry['sha256']:
            problems.append(entry['snapshot'])
    return problems


def audit():
    groups = defaultdict(lambda: dict(files=0, bytes=0))
    for parent, _dirs, files in walk(ROOT/'xarm_sdk_control/data'):
        for name in files:
            p = parent/name
            if p.is_symlink():
                continue
            relative = p.relative_to(ROOT/'xarm_sdk_control/data')
            group = groups[relative.parts[0]]
            group['files'] += 1
            group['bytes'] += p.stat().st_size
    caches = cache_paths()
    return dict(data_directories=dict(sorted(groups.items())),
                cache_paths=len(caches), cache_bytes=sum(map(file_size, caches)),
                broken_markdown_links=broken_links(), evidence_hash_errors=evidence_checks())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['audit', 'clean'])
    parser.add_argument('--apply', action='store_true', help='Actually remove generated Python/tool caches only')
    parser.add_argument('--output', type=Path, help='Write the JSON report here')
    args = parser.parse_args()
    if args.apply and args.action != 'clean':
        parser.error('--apply is only supported by clean')
    if args.action == 'audit':
        report = audit()
    else:
        paths = cache_paths()
        report = dict(applied=args.apply, reclaimed_bytes=sum(map(file_size, paths)),
                      paths=[p.relative_to(ROOT).as_posix() for p in paths])
        if args.apply:
            for p in paths:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
    text = json.dumps(report, ensure_ascii=False, indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')
    return int(bool(report.get('broken_markdown_links') or report.get('evidence_hash_errors')))


if __name__ == '__main__':
    raise SystemExit(main())
