"""Reliability of a small local LLM judge, using SQuAD gold answers as human labels.

Four probes, all with a 4-bit Qwen2.5-1.5B-Instruct judge run on CPU:
  1. pointwise agreement: is a gold answer / a wrong answer judged correct?
  2. position bias: pairwise (gold vs wrong) with the order swapped.
  3. length bias: gold vs a padded-but-still-correct version of the gold answer, and wrong vs padded wrong.
  4. prompt sensitivity: same pointwise question under two wordings.
The judge outputs a one-token verdict; we read its log-probabilities, so there is no sampling noise.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from .data import load_squad
from .metrics import bootstrap_ci

MODEL = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"


class Judge:
    def __init__(self, path=MODEL):
        from llama_cpp import Llama
        # logits_all stays False: we only need the last position, and the full matrix costs >1 GB.
        self.llm = Llama(model_path=path, n_ctx=1536, n_batch=256, n_threads=2, verbose=False)

    def choice_logit(self, prompt, options):
        """Return log-prob (renormalised over `options`) of each single-token option as the first reply token."""
        text = ("<|im_start|>system\nYou are a careful grader.<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
        toks = self.llm.tokenize(text.encode(), add_bos=False, special=True)
        self.llm.reset()
        self.llm.eval(toks)
        import llama_cpp
        ptr = llama_cpp.llama_get_logits_ith(self.llm._ctx.ctx, -1)
        row = np.ctypeslib.as_array(ptr, shape=(self.llm._n_vocab,)).astype(np.float64)
        row = row - row.max()
        logp = row - math.log(np.exp(row).sum())
        lp = {}
        for o in options:
            ids = self.llm.tokenize(o.encode(), add_bos=False)
            lp[o] = float(logp[ids[0]])
        z = math.log(sum(math.exp(v) for v in lp.values()) + 1e-30)
        return {k: v - z for k, v in lp.items()}


T1 = ("Passage:\n{ctx}\n\nQuestion: {q}\nProposed answer: {a}\n\n"
      "Is the proposed answer correct according to the passage? Reply with Yes or No.")
T2 = ("Read the text and check the answer.\n\nText: {ctx}\n\nQ: {q}\nA: {a}\n\n"
      "Does the text support this answer to the question? Reply with Yes or No.")
TP = ("Passage:\n{ctx}\n\nQuestion: {q}\n\nAnswer A: {a}\nAnswer B: {b}\n\n"
      "Which answer is correct according to the passage? Reply with A or B.")


def p_yes(judge, tmpl, ctx, q, a):
    lp = judge.choice_logit(tmpl.format(ctx=ctx, q=q, a=a), ["Yes", "No"])
    return math.exp(lp["Yes"])


def p_first(judge, ctx, q, a, b):
    lp = judge.choice_logit(TP.format(ctx=ctx, q=q, a=a, b=b), ["A", "B"])
    return math.exp(lp["A"])


def pad(ans, q):
    return f"Based on the passage, the answer to the question \"{q}\" is {ans}, as the text states."


def build_items(paragraphs, questions, n, seed):
    rng = random.Random(seed)
    qs = rng.sample(questions, n)
    by_para = {}
    for q in questions:
        by_para.setdefault(q.para_id, []).append(q)
    items = []
    for q in qs:
        gold = q.answers[0][2]
        others = [o for o in by_para[q.para_id] if o.answers[0][2].lower() != gold.lower()]
        pool = others or [o for o in questions if o.answers[0][2].lower() != gold.lower()]
        wrong = rng.choice(pool).answers[0][2]  # a real answer span from a different question
        items.append({"qid": q.qid, "q": q.text, "ctx": paragraphs[q.para_id], "gold": gold, "wrong": wrong})
    return items


def run(args):
    paragraphs, questions = load_squad(args.data)
    items = build_items(paragraphs, questions, args.n, args.seed)
    judge = Judge()
    out_path = Path(args.out)
    rows = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {r["qid"] for r in rows}
    t0 = time.time()
    for it in items:
        if it["qid"] in done:
            continue
        if args.max_items and len(rows) >= args.max_items:
            break
        c, q = it["ctx"], it["q"]
        r = {"qid": it["qid"], "gold": it["gold"], "wrong": it["wrong"]}
        r["yes_gold_t1"] = p_yes(judge, T1, c, q, it["gold"])
        r["yes_wrong_t1"] = p_yes(judge, T1, c, q, it["wrong"])
        r["yes_gold_t2"] = p_yes(judge, T2, c, q, it["gold"])
        r["yes_wrong_t2"] = p_yes(judge, T2, c, q, it["wrong"])
        r["pair_gold_first"] = p_first(judge, c, q, it["gold"], it["wrong"])
        r["pair_gold_second"] = 1 - p_first(judge, c, q, it["wrong"], it["gold"])
        r["yes_goldpad_t1"] = p_yes(judge, T1, c, q, pad(it["gold"], q))
        r["yes_wrongpad_t1"] = p_yes(judge, T1, c, q, pad(it["wrong"], q))
        r["pair_gold_vs_wrongpad"] = 1 - p_first(judge, c, q, pad(it["wrong"], q), it["gold"])
        r["pair_goldpad_vs_wrong"] = p_first(judge, c, q, pad(it["gold"], q), it["wrong"])
        rows.append(r)
        if len(rows) % 1 == 0:
            out_path.write_text(json.dumps(rows))
            print(len(rows), f"{time.time() - t0:.0f}s", flush=True)
    out_path.write_text(json.dumps(rows))


def summarise(rows):
    """Aggregate probes into point estimates with 95% bootstrap CIs."""
    a = lambda k: np.array([r[k] for r in rows])
    def ci(v):
        m, lo, hi = bootstrap_ci([float(x) for x in v])
        return {"mean": m, "lo": lo, "hi": hi}
    s = {"n": len(rows)}
    for t in ("t1", "t2"):
        s[f"acc_gold_{t}"] = ci(a(f"yes_gold_{t}") > 0.5)
        s[f"acc_wrong_{t}"] = ci(a(f"yes_wrong_{t}") < 0.5)
    s["pair_acc_gold_first"] = ci(a("pair_gold_first") > 0.5)
    s["pair_acc_gold_second"] = ci(a("pair_gold_second") > 0.5)
    s["pair_consistent_after_swap"] = ci((a("pair_gold_first") > 0.5) == (a("pair_gold_second") > 0.5))
    s["prompt_agreement_t1_t2"] = ci((a("yes_gold_t1") > 0.5) == (a("yes_gold_t2") > 0.5))
    s["yes_rate_gold_short"] = ci(a("yes_gold_t1") > 0.5)
    s["yes_rate_gold_padded"] = ci(a("yes_goldpad_t1") > 0.5)
    s["yes_rate_wrong_short"] = ci(a("yes_wrong_t1") > 0.5)
    s["yes_rate_wrong_padded"] = ci(a("yes_wrongpad_t1") > 0.5)
    s["pair_gold_beats_padded_wrong"] = ci(a("pair_gold_vs_wrongpad") > 0.5)
    s["pair_padded_gold_beats_wrong"] = ci(a("pair_goldpad_vs_wrong") > 0.5)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dev-v2.0.json")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/judge_raw.json")
    ap.add_argument("--max-items", type=int, default=None, help="stop after this many scored items (resumable)")
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()
    if args.summarise:
        rows = json.loads(Path(args.out).read_text())
        Path("results/judge_summary.json").write_text(json.dumps(summarise(rows), indent=1))
        print(json.dumps(summarise(rows), indent=1))
    else:
        run(args)


if __name__ == "__main__":
    main()
