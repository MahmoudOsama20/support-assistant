#!/usr/bin/env python3
"""Diagnostic table of failed cases from a results file (read-only; no tuning).

Run: python src/eval/show_failures.py results/eval/final-120b.json [--full CASE_ID]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    p.add_argument("--full", help="dump the full payload and detail of one case")
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    data = json.loads(args.path.read_text(encoding="utf-8"))
    rows = data["cases"]
    if args.full:
        row = next(r for r in rows if r["id"] == args.full)
        print(json.dumps({"query": row["query"], "payload": row["payload"], "detail": row["detail"]},
                         ensure_ascii=False, indent=1))
        return
    print(f"{'id':<22}{'exp->got':<20}{'route':<15}{'conf':>6} {'reason':<14}{'drop':>5}  cited  checks_failed")
    for r in rows:
        if r["passed"]:
            continue
        pl, det = r["payload"], r.get("detail") or {}
        cited = ",".join(sorted({c.get("doc_id", "?") for c in pl.get("citations") or []})) or "-"
        failed = [k for k, v in r["checks"].items() if not v]
        conf = pl.get("route_confidence")
        print(f"{r['id']:<22}{r['expected_status'] + '->' + str(r['status']):<20}{str(r['route']):<15}"
              f"{(f'{conf:.3f}' if conf is not None else '-'):>6} {str(r.get('reason')):<14}"
              f"{len(det.get('dropped_claims') or []):>5}  {cited}  {failed}")


if __name__ == "__main__":
    main()