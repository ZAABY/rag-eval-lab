"""Retrievers: BM25, dense embeddings, reciprocal-rank-fusion hybrid, cross-encoder rerank."""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

TOKEN = re.compile(r"[a-z0-9]+")
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


def tokenize(text):
    return TOKEN.findall(text.lower())


class BM25Retriever:
    def __init__(self, texts):
        self.index = BM25Okapi([tokenize(t) for t in texts])

    def scores(self, query):
        return np.asarray(self.index.get_scores(tokenize(query)), dtype=np.float32)


class DenseRetriever:
    def __init__(self, texts, batch_size=8):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(DENSE_MODEL)
        emb = np.asarray(list(self.model.embed(texts, batch_size=batch_size)), dtype=np.float32)
        self.emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    def query_matrix(self, queries):
        q = np.asarray(list(self.model.query_embed(queries)), dtype=np.float32)
        return q / np.linalg.norm(q, axis=1, keepdims=True)

    def scores_matrix(self, queries):
        return self.query_matrix(queries) @ self.emb.T


def top_ids(scores, k):
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx], kind="stable")].tolist()


def rrf(rankings, k=60, top=100):
    """Reciprocal rank fusion of several ranked id lists."""
    fused = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(fused, key=lambda c: (-fused[c], c))[:top]


class Reranker:
    def __init__(self):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(RERANK_MODEL)

    def rerank(self, query, texts, ids):
        scores = list(self.model.rerank(query, texts))
        order = sorted(range(len(ids)), key=lambda i: (-scores[i], i))
        return [ids[i] for i in order]
