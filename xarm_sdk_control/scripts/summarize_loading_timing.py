#!/usr/bin/env python3
"""Summarize saved loading result.json files offline; no camera/robot connection."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from xarm_sdk_control.loading_performance import render_loading_report, summarize_loading_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="Saved result.json paths; keep failed runs too.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for JSON and Markdown.")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("--output-dir must be a new directory")
    paths = [path.resolve() for path in args.results]
    if len(set(paths)) != len(paths):
        parser.error("Duplicate result paths would count the same run twice")
    reports = []
    for path in paths:
        try:
            raw = path.read_bytes()
            report = summarize_loading_result(json.loads(raw), str(path))
            report["source_sha256"] = hashlib.sha256(raw).hexdigest()
            reports.append(report)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f"Cannot summarize {path}: {exc}")
    markdown = render_loading_report(reports)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "report.json").write_text(
        json.dumps({"schema_version": 1, "runs": reports}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Saved: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
