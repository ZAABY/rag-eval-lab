"""Run the retrieval study and write per-question metrics and a summary table."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import metrics as M
from .data import chunk_paragraphs, index_by_para, load_squad, relevant_chunks, sample_questions
from .retrieval import BM25Retriever, DenseRetriever, Reranker, rrf, top_ids

CHUNKINGS = {"paragraph": None, "w120": 120, "w60": 60, "w30": 30}
METHODS = ["bm25", "dense", "hybrid", "hybrid+rerank"]


def evaluate(paragraphs, questions, chunking, reranker, pool=20):
    chunks = chunk_paragraphs(paragraphs, CHUNKINGS[chunking])
    by_para = index_by_para(chunks)
    texts = [c.text for c in chunks]
    bm25 = BM25Retriever(texts)
    dense = DenseRetriever(texts)
    qtexts = [q.text for q in questions]
    dense_scores = dense.scores_matrix(qtexts)
    per_q = {m: [] for m in METHODS}
    for qi, q in enumerate(questions):
        rel = relevant_chunks(q, by_para)
        if not rel:  # answer straddles every chunk boundary; cannot be scored fairly
            continue
        b = top_ids(bm25.scores(q.text), 100)
        d = top_ids(dense_scores[qi], 100)
        h = rrf([b, d])
        rr = reranker.rerank(q.text, [texts[i] for i in h[:pool]], h[:pool]) + h[pool:]
        for name, ranked in (("bm25", b), ("dense", d), ("hybrid", h), ("hybrid+rerank", rr)):
            per_q[name].append({
                "qid": q.qid,
                "r1": M.recall_at_k(ranked, rel, 1),
                "r5": M.recall_at_k(ranked, rel, 5),
                "r10": M.recall_at_k(ranked, rel, 10),
                "mrr10": M.mrr_at_k(ranked, rel, 10),
                "ndcg10": M.ndcg_at_k(ranked, rel, 10),
            })
    return {"n_chunks": len(chunks), "n_scored": len(per_q["bm25"]), "per_question": per_q}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dev-v2.0.json")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunkings", nargs="+", default=list(CHUNKINGS))
    ap.add_argument("--out", default="results/retrieval.json")
    args = ap.parse_args()
    paragraphs, questions = load_squad(args.data)
    sample = sample_questions(questions, args.n, args.seed)
    reranker = Reranker()
    out = {"n_requested": args.n, "seed": args.seed, "results": {}}
    for ch in args.chunkings:
        t = time.time()
        res = evaluate(paragraphs, sample, ch, reranker)
        out["results"][ch] = res
        print(ch, res["n_chunks"], res["n_scored"], f"{time.time() - t:.0f}s", flush=True)
        Path(args.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
