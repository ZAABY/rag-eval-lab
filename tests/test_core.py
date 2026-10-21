import math

from ragevallab.data import Question, chunk_paragraphs, index_by_para, relevant_chunks
from ragevallab.metrics import bootstrap_ci, mrr_at_k, ndcg_at_k, paired_bootstrap_diff, recall_at_k


def test_chunks_cover_text_and_offsets_match():
    text = "one two three four five six seven"
    chunks = chunk_paragraphs([text], 3)
    assert [c.text for c in chunks] == ["one two three", "four five six", "seven"]
    for c in chunks:
        assert text[c.start:c.end] == c.text


def test_relevance_requires_full_answer_span():
    text = "alpha beta gamma delta epsilon"
    chunks = chunk_paragraphs([text], 2)  # alpha beta | gamma delta | epsilon
    by_para = index_by_para(chunks)
    start = text.index("beta gamma")
    q = Question("q", 0, "?", ((start, start + len("beta gamma"), "beta gamma"),))
    assert relevant_chunks(q, by_para) == set()  # span straddles two chunks
    q2 = Question("q", 0, "?", ((text.index("delta"), text.index("delta") + 5, "delta"),))
    assert relevant_chunks(q2, by_para) == {1}


def test_metrics_known_values():
    ranked = [5, 7, 9]
    assert recall_at_k(ranked, {9}, 2) == 0.0
    assert recall_at_k(ranked, {9}, 3) == 1.0
    assert mrr_at_k(ranked, {7}) == 0.5
    assert math.isclose(ndcg_at_k([1, 2], {2}), 1 / math.log2(3))
    assert ndcg_at_k([2, 1], {2}) == 1.0


def test_bootstrap_ci_brackets_mean_and_is_seeded():
    vals = [0.0, 1.0] * 50
    m, lo, hi = bootstrap_ci(vals, n_boot=500, seed=1)
    assert lo <= m <= hi and (m, lo, hi) == bootstrap_ci(vals, n_boot=500, seed=1)
    m2, lo2, hi2 = paired_bootstrap_diff([1.0] * 40, [0.0] * 40, n_boot=200)
    assert m2 == 1.0 and lo2 == hi2 == 1.0
