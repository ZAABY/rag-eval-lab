# rag-eval-lab

A small, reproducible study of two questions, run entirely on free open models and a 2-CPU machine:

1. **Retrieval:** how much do chunk size and retrieval method (BM25, dense, hybrid, hybrid + reranker) change whether a RAG pipeline finds the passage that contains the answer?
2. **Judge reliability (add-on):** how far can you trust a small local LLM (Qwen2.5-1.5B-Instruct, 4-bit) as an automatic answer judge? SQuAD gold answers act as the human labels.

Everything here is measured, with 95% bootstrap confidence intervals. Limits are listed at the bottom. Read them before quoting any number.

## Setup

- Data: SQuAD v2 dev, answerable questions only (35 articles, 1,204 paragraphs, 5,928 questions). A random seeded sample is scored.
- Chunking: word windows of 30, 60 and 120 words, no overlap. A chunk is relevant if it fully contains a gold answer span.
- Retrievers: BM25 (`rank-bm25`), dense (`BAAI/bge-small-en-v1.5` via fastembed), hybrid (reciprocal rank fusion of the two), hybrid + cross-encoder rerank of the top 20 (`Xenova/ms-marco-MiniLM-L-6-v2`).
- Judge: `Qwen2.5-1.5B-Instruct` Q4_K_M via llama.cpp. It answers with one token (Yes/No or A/B). We read its log-probabilities, so there is no sampling noise.

## Retrieval results (300 questions per chunking)

| chunking | method | recall@1 | MRR@10 |
|---|---|---|---|
| 30 words | BM25 | 0.579 | 0.660 |
| 30 words | hybrid + rerank | 0.717 | 0.774 |
| 60 words | hybrid + rerank | 0.805 | 0.855 |
| 120 words | hybrid + rerank | 0.883 | 0.919 |

Full table with confidence intervals: [`results/retrieval_summary.md`](results/retrieval_summary.md).

What the data supports:

- Adding the cross-encoder reranker helped at every chunk size. Paired MRR@10 gain over plain hybrid: +0.066 (30 words), +0.065 (60), +0.096 (120). All three intervals exclude zero.
- Hybrid beat BM25 alone at 30 and 60 words (+0.049, +0.044, intervals exclude zero). At 120 words the gain was +0.015 and the interval includes zero, so this study cannot say hybrid helps there.
- Dense retrieval alone was the weakest method at 60 and 120 words. SQuAD questions reuse many words from the passage, which favours lexical matching. This will not hold for every corpus.

Do not compare scores across chunk sizes as "bigger chunks are better retrievers". A larger chunk contains the answer more easily and there are fewer chunks to pick from, so the task itself gets easier.

## Judge reliability results (41 questions)

Each question is paired with its gold answer and a wrong answer (a real answer span from another question in the same paragraph).

| probe | result | 95% CI |
|---|---|---|
| says "correct" to the gold answer (prompt 1) | 0.83 | 0.71 - 0.93 |
| says "incorrect" to the wrong answer (prompt 1) | 0.88 | 0.78 - 0.98 |
| pairwise pick gold, gold shown first | 0.93 | 0.85 - 1.00 |
| pairwise pick gold, gold shown second | 0.95 | 0.88 - 1.00 |
| same verdict on both wordings of the pointwise prompt | 0.88 | 0.76 - 0.98 |
| accepts the wrong answer when short | 0.12 | 0.02 - 0.22 |
| accepts the same wrong answer padded with "Based on the passage, the answer is ..." | 0.41 | 0.27 - 0.56 |

What this shows:

- **Verbosity bias is the clearest effect.** Wrapping a wrong answer in confident, longer wording raised its acceptance rate from 12% to 41%. The padded text adds no information.
- **Pointwise verdicts depend on the prompt.** The two wordings disagreed on about 1 in 8 gold answers.
- **Position bias was small here.** Pairwise accuracy was similar in both orders, and the choice survived a swap for 98% of questions.

These are results for one 1.5B model. They say nothing about frontier-model judges.

## Limits

- Only 41 judge questions and 300 retrieval questions per chunking. Intervals are wide. Small differences are not resolved.
- The wrong answers in the judge study are easy negatives (real spans from the same paragraph, often a different type). A harder, near-miss negative set would lower judge accuracy.
- A 1.5B model on CPU. Larger judges may behave differently.
- SQuAD is one-paragraph extractive QA. Real RAG corpora are messier. Results may not transfer.
- A whole-paragraph chunking was planned but dropped: it ran out of memory repeatedly on the 2 GB machine, so only 30/60/120-word windows are reported.
- No overlap between chunks, one fixed reranker pool size (20), one embedding model.

## Reproduce

```
uv venv && uv pip install -e .
python -m ragevallab.run_retrieval --n 300 --chunkings w60 --out results/retrieval_w60.json
python -m ragevallab.summarize_retrieval
python -m ragevallab.judge --n 120 --max-items 41 --out results/judge_raw.json   # resumable
python -m ragevallab.judge --summarise
```

The judge model file (`qwen2.5-1.5b-instruct-q4_k_m.gguf`) and the SQuAD dev file are downloaded separately from Hugging Face and the SQuAD site and are not committed.
