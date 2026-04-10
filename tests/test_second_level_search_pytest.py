from Matcher.pipeline.trial_search.second_level_search import (
    SecondStageRetriever,
    nct_ids_without_criterion_hits,
)


class DummyES:
    def search(self, *args, **kwargs):
        return {"hits": {"hits": []}}


def test_score_criteria_without_llm_weights():
    retriever = SecondStageRetriever(
        es_client=DummyES(),
        llm_reranker=None,
        embedder=None,
        index_name="idx",
        inclusion_weight=1.0,
        exclusion_weight=0.25,
    )
    criteria = [
        {"_score": 2.0, "_source": {"eligibility_type": "Inclusion Criteria"}},
        {"_score": 1.0, "_source": {"eligibility_type": "Exclusion Criteria"}},
    ]
    scored = retriever.score_criteria_without_llm(criteria)
    assert scored[0]["llm_score"] == 1.0
    assert scored[1]["llm_score"] == 0.125


def test_nct_ids_without_criterion_hits():
    candidates = ["NCT0001", "NCT0002", "nct0003"]
    criteria = [
        {"_source": {"nct_id": "NCT0001"}},
        {"_source": {"nct_id": "nct0003"}},
    ]
    missing = nct_ids_without_criterion_hits(candidates, criteria)
    assert missing == ["NCT0002"]


def test_aggregate_to_trials_weighted():
    retriever = SecondStageRetriever(
        es_client=DummyES(),
        llm_reranker=None,
        embedder=None,
        index_name="idx",
    )
    criteria = [
        {"llm_score": 0.6, "_source": {"nct_id": "N1"}},
        {"llm_score": 0.5, "_source": {"nct_id": "N1"}},
        {"llm_score": 0.9, "_source": {"nct_id": "N2"}},
    ]
    trials = retriever.aggregate_to_trials(criteria, threshold=0.5, method="weighted")
    assert trials[0]["nct_id"] == "N2"
