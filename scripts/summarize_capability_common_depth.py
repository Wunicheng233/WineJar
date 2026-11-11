#!/usr/bin/env python3
"""Summarize identical valid-pixel comparisons after both saved depth audits.

Camera errors are pooled within a jar first. Reported percentiles are across
jar cases, not a claim that pixels, cameras or paired layouts are independent.
"""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def summarize(first, second, output):
    roots, output = [Path(first), Path(second)], Path(output)
    if output.exists():
        raise FileExistsError(output)
    sources = {}
    for root in roots:
        for name in ("measurement_metrics.json", "validation.json", "provenance.json"):
            path = root/name
            sources[str(path.resolve())] = sha256(path.read_bytes()).hexdigest()
        if not read(root/"validation.json")["passed"]:
            raise ValueError("source_evaluation_not_validated:"+str(root))
    tables = [{r["pair_id"]:r for r in read(root/"measurement_metrics.json")} for root in roots]
    if not tables[0] or tables[0].keys() != tables[1].keys():
        raise ValueError("comparison_pair_sets_differ")
    grouped = defaultdict(list)
    checked_regions = 0
    for pair_id in sorted(tables[0]):
        rows = [table[pair_id] for table in tables]
        if any(row["status"] != "ok" for row in rows):
            raise ValueError("source_measurement_failed:"+pair_id)
        if rows[0]["regions"].keys() != rows[1]["regions"].keys():
            raise ValueError("comparison_region_sets_differ")
        for region in rows[0]["regions"]:
            metrics = [row["regions"][region] for row in rows]
            common = [m["common_valid_comparison"] for m in metrics]
            if any(c["status"] != "ok" for c in common):
                raise ValueError("comparison_intersection_failed:"+pair_id)
            if (common[0]["common_valid_pixels"] != common[1]["common_valid_pixels"] or
                common[0]["fixed_reference_pixels"] != common[1]["fixed_reference_pixels"] or
                set(common[0]["members"]) != set(common[1]["members"])):
                raise ValueError("comparison_intersection_disagrees:"+pair_id)
            checked_regions += 1
            for method, (row, own, shared) in enumerate(zip(rows, metrics, common)):
                grouped[(row["layout"], row["condition"], row["case"], region, method)].append((own, shared))
    cases = []
    for (layout, condition, case, region, method), metrics in sorted(grouped.items()):
        if len(metrics) != 3:
            raise ValueError("expected_three_cameras_per_case")
        reference = sum(shared["fixed_reference_pixels"] for _, shared in metrics)
        common_count = sum(shared["common_valid_pixels"] for _, shared in metrics)
        own_valid = sum(own["valid_prediction_pixels"] for own, _ in metrics)
        values = [(shared["common_valid_pixels"], shared["rmse_mm"], shared["signed_mean_mm"])
                  for _, shared in metrics if shared["common_valid_pixels"]]
        cases.append(dict(layout=layout, condition=condition, case=case, region=region, method=method,
            reference_pixels=reference, common_pixels=common_count,
            common_coverage=common_count/reference if reference else None,
            own_coverage=own_valid/reference if reference else None,
            common_rmse_mm=(float(np.sqrt(sum(n*rmse**2 for n, rmse, _ in values)/common_count))
                            if common_count else None),
            common_signed_mean_mm=(sum(n*mean for n, _, mean in values)/common_count if common_count else None)))
    summaries = []
    keys = sorted({(r["layout"], r["condition"], r["region"], r["method"]) for r in cases})
    for key in keys:
        rows = [r for r in cases if (r["layout"], r["condition"], r["region"], r["method"]) == key]
        result = dict(zip(("layout", "condition", "region", "method"), key), jar_cases=len(rows))
        for metric in ("common_rmse_mm", "common_signed_mean_mm", "own_coverage", "common_coverage"):
            values = [r[metric] for r in rows if r[metric] is not None]
            for suffix, percentile in (("median", 50), ("p95", 95), ("max", 100)):
                result[metric+"_"+suffix] = float(np.percentile(values, percentile)) if values else None
        summaries.append(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(schema=1, validation=dict(passed=True,
        matched_pairs=len(tables[0]), matched_region_intersections=checked_regions),
        methods=[str(root.resolve()) for root in roots], input_sha256=sources,
        source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        aggregation="pool squared/signed error over three cameras within each case; percentiles across cases",
        limitations="previously viewed simulation; paired layouts share jar geometry; no independence or confidence-interval claim",
        summary=summaries, case_metrics=cases), indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.first, args.second, args.output)
