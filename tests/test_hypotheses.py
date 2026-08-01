from semadexp.data.corpora import generate_feedback_corpus
from semadexp.hypothesis.evaluate import evaluate_generation
from semadexp.hypothesis.generate import extract_candidate_hypotheses, score_priorities
from semadexp.hypothesis.knowledge_base import build_knowledge_base
from semadexp.llm.client import DeterministicLLM


def test_hypothesis_pipeline_quantified():
    kb = build_knowledge_base(10, seed=5, n_advertisers=40, days=3, sessions_per_day=200)
    corpus = generate_feedback_corpus(seed=5)
    hyps = extract_candidate_hypotheses(corpus, DeterministicLLM())
    ranked = score_priorities(hyps, kb)
    assert len(ranked) > 0
    priorities = [h.priority for h in ranked]
    assert priorities == sorted(priorities, reverse=True)
    assert max(priorities) > 0 or min(priorities) < 0
    rep = evaluate_generation(ranked, kb, seed=5)
    assert 0.0 <= rep.recall_at_top_x <= 1.0
    assert len(rep.top_k_capture) > 0
