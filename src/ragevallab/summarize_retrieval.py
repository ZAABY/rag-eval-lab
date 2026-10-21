"""Turn results/retrieval_<chunking>.json into a markdown table with 95% bootstrap CIs and paired differences."""
import json
from pathlib import Path

from .metrics import bootstrap_ci, paired_bootstrap_diff

METHODS = ["bm25", "dense", "hybrid", "hybrid+rerank"]


def main():
    lines = ["| chunking | chunks | scored q | method | recall@1 | recall@5 | MRR@10 | nDCG@10 |", "|---|---|---|---|---|---|---|---|"]
    diffs = []
    for c in ["w30", "w60", "w120"]:
        p = Path(f"results/retrieval_{c}.json")
        if not p.exists():
            continue
        r = json.loads(p.read_text())["results"][c]
        for m in METHODS:
            rows = r["per_question"][m]
            cells = []
            for k in ("r1", "r5", "mrr10", "ndcg10"):
                mean, lo, hi = bootstrap_ci([x[k] for x in rows])
                cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]")
            lines.append(f"| {c} | {r['n_chunks']} | {r['n_scored']} | {m} | " + " | ".join(cells) + " |")
        pq = r["per_question"]
        for a, b in (("hybrid", "bm25"), ("hybrid", "dense"), ("hybrid+rerank", "hybrid")):
            d = paired_bootstrap_diff([x["mrr10"] for x in pq[a]], [x["mrr10"] for x in pq[b]])
            diffs.append(f"| {c} | {a} - {b} | " + " | ".join(f"{v:.3f}" for v in d) + " |")
    out = "\n".join(lines) + "\n\nPaired MRR@10 differences (mean, 95% CI low, high):\n\n| chunking | pair | mean | lo | hi |\n|---|---|---|---|---|\n" + "\n".join(diffs)
    Path("results/retrieval_summary.md").write_text(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
