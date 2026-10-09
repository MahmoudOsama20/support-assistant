#!/usr/bin/env python3
"""Retrieval comparison + gate-signal dump on the calibration queries (a DEV set; never the test set).

Run: python src/rag/retrieval_eval.py [--device auto|cpu|cuda] [--no-stem] [--rerank-top 20]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

if __package__ in (None, ""):  # direct execution: make `rag` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.bm25 import BM25Index  # noqa: E402
from rag.chunking import load_chunks  # noqa: E402
from rag.dense import DEFAULT_CACHE, BgeM3Encoder, DenseIndex  # noqa: E402
from rag.metrics import DEFAULT_CASES, REPO_ROOT, auc, first_gold_rank, load_cases, summarize  # noqa: E402
from rag.rerank import CrossEncoderReranker, sigmoid  # noqa: E402
from rag.retriever import HybridRetriever  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "results" / "rag" / "retrieval_calibration.json"
ARMS = ["bm25_nostem", "bm25_stem", "dense", "hybrid", "hybrid_rerank"]
DEPTH = 20
SIGNALS = {"rerank_logit": "rerank_logit", "dense_cosine": "dense_top1_cosine",
           "bm25_score": "bm25_top1", "rrf_score": "rrf_top1"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--no-stem", action="store_true", help="hybrid arms use BM25 without light stemming")
    p.add_argument("--rerank-top", type=int, default=20)
    p.add_argument("--index-cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    chunks = load_chunks()
    cases = load_cases(args.cases, {c.doc_id for c in chunks})
    answerable = [c for c in cases if c.answerable]
    if not answerable:
        raise SystemExit("no answerable cases in the file")

    bm25_stem = BM25Index(chunks, light_stem=True)
    bm25_nostem = BM25Index(chunks, light_stem=False)
    encoder = BgeM3Encoder(device=args.device)
    dense = DenseIndex(chunks, encoder, cache_path=args.index_cache)
    reranker = CrossEncoderReranker(device=args.device)
    retriever = HybridRetriever(bm25_nostem if args.no_stem else bm25_stem, dense, reranker,
                                fetch_k=DEPTH, rerank_top=args.rerank_top)
    retriever.retrieve("warm up")  # excluded from timings

    ranks: dict[str, dict[str, int | None]] = {arm: {} for arm in ARMS}
    timings: dict[str, list[float]] = {}
    records: list[dict] = []
    for case in cases:
        res = retriever.retrieve(case.query, final_k=DEPTH)
        assert res.reranked, "reranker returned nothing"
        arms = {"bm25_nostem": bm25_nostem.search(case.query, DEPTH),
                "bm25_stem": bm25_stem.search(case.query, DEPTH),
                "dense": res.dense, "hybrid": res.fused, "hybrid_rerank": res.reranked}
        for stage, ms in res.timings_ms.items():
            timings.setdefault(stage, []).append(ms)
        case_ranks = None
        if case.answerable:
            gold = set(case.gold_doc_ids)
            case_ranks = {arm: first_gold_rank([h.chunk.doc_id for h in hits], gold)
                          for arm, hits in arms.items()}
            for arm, r in case_ranks.items():
                ranks[arm][case.id] = r
        top = res.reranked[0]
        records.append({
            "id": case.id, "category": case.category, "language": case.language,
            "answerable": case.answerable, "gold_doc_ids": list(case.gold_doc_ids),
            "top1_chunk": top.chunk.chunk_id, "rerank_logit": top.score, "rerank_prob": sigmoid(top.score),
            "dense_top1_cosine": res.dense[0].score,
            "bm25_top1": res.bm25[0].score if res.bm25 else 0.0,
            "rrf_top1": res.fused[0].score,
            "ranks": case_ranks,
            "top5_chunks": {arm: [h.chunk.chunk_id for h in hits[:5]] for arm, hits in arms.items()},
        })

    print(f"\nRetrieval on {len(answerable)} answerable queries "
          f"(chunk-level rank of first gold doc, depth {DEPTH}; hybrid BM25 stemming "
          f"{'OFF' if args.no_stem else 'ON'})")
    print(f"{'arm':<15}{'R@1':>7}{'R@3':>7}{'R@5':>7}{'MRR':>7}")
    for arm in ARMS:
        s = summarize([ranks[arm][c.id] for c in answerable])
        print(f"{arm:<15}{s['recall@1']:>7.3f}{s['recall@3']:>7.3f}{s['recall@5']:>7.3f}{s['mrr']:>7.3f}")

    cats = sorted({c.category for c in answerable})
    counts = {cat: sum(c.category == cat for c in answerable) for cat in cats}
    print("\nMRR by category (tiny n, noisy): " + ", ".join(f"{cat} n={counts[cat]}" for cat in cats))
    print(f"{'arm':<15}" + "".join(f"{cat:>10}" for cat in cats))
    for arm in ARMS:
        cells = [summarize([ranks[arm][c.id] for c in answerable if c.category == cat])["mrr"] for cat in cats]
        print(f"{arm:<15}" + "".join(f"{v:>10.2f}" for v in cells))

    misses = [r for r in records if r["answerable"]
              and (r["ranks"]["hybrid_rerank"] is None or r["ranks"]["hybrid_rerank"] > 3)]
    print(f"\nhybrid_rerank misses (gold not in top 3): {len(misses)}")
    for r in misses:
        print(f"  {r['id']} {r['category']} rank={r['ranks']['hybrid_rerank']} gold={r['gold_doc_ids']} "
              f"top3={r['top5_chunks']['hybrid_rerank'][:3]}")

    positives = [r for r in records if r["answerable"]]
    negatives = [r for r in records if not r["answerable"]]
    if negatives:
        print(f"\nGate signals: answerable n={len(positives)} vs unanswerable n={len(negatives)}")
        for name, key in SIGNALS.items():
            value = auc([r[key] for r in positives], [r[key] for r in negatives])
            print(f"  AUC {name:<14}{value:.3f}")
    print(f"\n{'id':<8}{'label':<7}{'p_rerank':>9}{'cos':>8}{'bm25':>8}  top1_chunk (sorted by p_rerank)")
    for r in sorted(records, key=lambda r: -r["rerank_prob"]):
        label = "ANS" if r["answerable"] else "UNANS"
        print(f"{r['id']:<8}{label:<7}{r['rerank_prob']:>9.4f}{r['dense_top1_cosine']:>8.3f}"
              f"{r['bm25_top1']:>8.2f}  {r['top1_chunk']}")

    device = f"{encoder.device} (fp16)" if encoder.device.type == "cuda" else f"{encoder.device} (fp32)"
    print(f"\nPer-stage latency on this machine, device {device}, n={len(cases)} (warm-up excluded):")
    for stage, values in timings.items():
        print(f"  {stage:<8} p50 {statistics.median(values):8.1f} ms   max {max(values):8.1f} ms")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {"device": str(encoder.device), "encoder": encoder.name, "reranker": reranker.name,
                   "hybrid_stemming": not args.no_stem, "rerank_top": args.rerank_top, "depth": DEPTH,
                   "cases_file": str(args.cases.name)},
        "records": records}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()