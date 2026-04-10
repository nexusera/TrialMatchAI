import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Set

from Matcher.models.embedding.text_embedder import TextEmbedder
from Matcher.models.llm.llm_reranker import LLMReranker
from Matcher.utils.file_utils import write_text_file
from Matcher.utils.logging_config import setup_logging
from Matcher.utils.retry import with_retries

from elasticsearch import Elasticsearch

logger = setup_logging(__name__)

# Elasticsearch caps total terms across all `terms` queries in one request at
# index.max_terms_count (default 65536). Splitting with bool.should still counts
# every term, so large first-stage result sets must use multiple searches.
_MAX_NCT_TERMS_PER_CLAUSE = 60000


def _nct_id_filter_clause(nct_ids: List[str]) -> Dict:
    """Filter by nct_id. Empty list matches nothing. Non-empty list must be
    length <= _MAX_NCT_TERMS_PER_CLAUSE (callers batch larger lists)."""
    if not nct_ids:
        return {"bool": {"must_not": {"match_all": {}}}}
    if len(nct_ids) > _MAX_NCT_TERMS_PER_CLAUSE:
        raise ValueError(
            f"nct_id filter has {len(nct_ids)} terms; max {_MAX_NCT_TERMS_PER_CLAUSE}"
        )
    return {"terms": {"nct_id": nct_ids}}


def _merge_hits_by_score(hits: List[Dict], size: int) -> List[Dict]:
    hits.sort(key=lambda h: float(h.get("_score", 0.0)), reverse=True)
    return hits[:size]


def nct_ids_without_criterion_hits(
    candidate_nct_ids: List[str], all_criteria: List[Dict]
) -> List[str]:
    """NCT IDs from the candidate list that never appear on any criterion hit."""
    seen: Set[str] = set()
    for hit in all_criteria:
        src = hit.get("_source")
        if not isinstance(src, dict):
            continue
        nid = src.get("nct_id")
        if nid is None:
            continue
        key = str(nid).strip().upper()
        if key:
            seen.add(key)
    missing: List[str] = []
    missing_keys: Set[str] = set()
    for nid in candidate_nct_ids:
        key = str(nid).strip().upper()
        if not key or key in seen or key in missing_keys:
            continue
        missing_keys.add(key)
        missing.append(str(nid).strip())
    return missing


class SecondStageRetriever:
    def __init__(
        self,
        es_client: Elasticsearch,
        llm_reranker: Optional[LLMReranker],  # Make optional
        embedder: Optional[TextEmbedder],
        index_name: str,
        size: int = 250,
        inclusion_weight: float = 1.0,
        exclusion_weight: float = 0.25,
        bio_med_ner=None,
        search_mode: str = "hybrid",
        second_level_vector_score_threshold: float = 0.5,
        second_level_aggregate_score_threshold: float = 0.5,
    ):
        self.es_client = es_client
        self.llm_reranker = llm_reranker  # Can be None
        self.embedder = embedder
        self.index_name = index_name
        self.size = size
        self.inclusion_weight = inclusion_weight
        self.exclusion_weight = exclusion_weight
        self.bio_med_ner = bio_med_ner
        self.search_mode = search_mode.lower() if search_mode else "hybrid"
        self.second_level_vector_score_threshold = float(
            second_level_vector_score_threshold
        )
        self.second_level_aggregate_score_threshold = float(
            second_level_aggregate_score_threshold
        )

    def retrieve_all_criteria(self, nct_ids: List[str]) -> List[Dict]:
        """Load all criteria documents for the candidate trials without query-time retrieval."""
        if not nct_ids:
            return []

        filters = [
            _nct_id_filter_clause(nct_ids[i : i + _MAX_NCT_TERMS_PER_CLAUSE])
            for i in range(0, len(nct_ids), _MAX_NCT_TERMS_PER_CLAUSE)
        ]
        all_hits: List[Dict] = []
        page_size = 1000

        for filt in filters:
            from_offset = 0
            while True:
                body = {
                    "from": from_offset,
                    "size": page_size,
                    "query": {"bool": {"filter": filt}},
                }
                response = with_retries(
                    lambda b=body: self.es_client.search(index=self.index_name, body=b),
                    logger=logger,
                    action="ES criteria full scan",
                )
                hits = response["hits"]["hits"]
                all_hits.extend(hits)
                if len(hits) < page_size:
                    break
                from_offset += page_size

        logger.info(
            "[%s] Loaded %s criteria documents across %s candidate trials",
            self.search_mode,
            len(all_hits),
            len(nct_ids),
        )
        return all_hits

    def _search_criteria_batched(
        self,
        nct_ids: List[str],
        build_query: Callable[[Dict], Dict],
        action: str,
    ) -> List[Dict]:
        """Run one ES search per nct_id chunk so total terms stay under max_terms_count."""
        if not nct_ids:
            filters: List[Dict] = [_nct_id_filter_clause([])]
        else:
            filters = [
                _nct_id_filter_clause(
                    nct_ids[i : i + _MAX_NCT_TERMS_PER_CLAUSE]
                )
                for i in range(0, len(nct_ids), _MAX_NCT_TERMS_PER_CLAUSE)
            ]
        n_batches = len(filters)
        fetch_size = (
            self.size
            if n_batches <= 1
            else max(self.size, min(5000, self.size * n_batches))
        )
        all_hits: List[Dict] = []
        for filt in filters:
            body = {"size": fetch_size, "query": build_query(filt)}
            response = with_retries(
                lambda b=body: self.es_client.search(
                    index=self.index_name, body=b
                ),
                logger=logger,
                action=action,
            )
            all_hits.extend(response["hits"]["hits"])
        return _merge_hits_by_score(all_hits, self.size)

    def get_synonyms(self, condition: str) -> List[str]:
        if self.bio_med_ner is None:
            logger.warning("BioMedNER not initialized; cannot extract synonyms.")
            return []
        try:
            raw_result = self.bio_med_ner.annotate_texts_in_parallel(
                [condition], max_workers=1
            )
            ner_results = raw_result
            if ner_results and ner_results[0]:
                synonyms = set()
                for entity in ner_results[0]:
                    if isinstance(entity, dict):
                        if entity.get("entity_group", "").lower() == "disease":
                            for syn in entity.get("synonyms", []):
                                if isinstance(syn, str) and syn.strip():
                                    synonyms.add(syn.strip())
                    elif isinstance(entity, str) and entity.strip():
                        # Some BioMedNER setups may return plain strings.
                        synonyms.add(entity.strip())
                return list(synonyms)
            logger.warning(f"No annotations found for condition: {condition}")
        except Exception as exc:
            logger.error(
                "BioMedNER synonym extraction failed for '%s': %s", condition, exc
            )
        return []

    def retrieve_criteria(
        self, nct_ids: List[str], queries: List[str]
    ) -> Dict[str, List[Dict]]:
        query_to_hits = {}

        def execute_query(query):
            # Use entities.synonyms only if BioMedNER is enabled
            fields_to_search = (
                ["criterion", "entities.synonyms"]
                if self.bio_med_ner is not None
                else ["criterion"]
            )

            def build_bm25_query(nct_filter: Dict) -> Dict:
                return {
                    "bool": {
                        "should": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": fields_to_search,
                                    "type": "best_fields",
                                    "operator": "and",
                                }
                            },
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": fields_to_search,
                                    "type": "phrase",
                                }
                            },
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": fields_to_search,
                                    "type": "best_fields",
                                    "operator": "or",
                                }
                            },
                        ],
                        "minimum_should_match": 1,
                        "filter": nct_filter,
                    }
                }

            if self.search_mode == "bm25":
                try:
                    hits = self._search_criteria_batched(
                        nct_ids, build_bm25_query, "ES criteria search"
                    )
                    logger.info(
                        "[%s] Retrieved %s documents for query: '%s'",
                        self.search_mode,
                        len(hits),
                        query,
                    )
                    return query, hits
                except Exception:
                    logger.exception(
                        "Second-level search failed for query: %s", query
                    )
                    return query, []

            if self.search_mode == "vector":
                if self.embedder is None:
                    logger.warning(
                        "Vector mode selected but embedder is None. Falling back to BM25."
                    )
                    return execute_query_bm25(query)

                vectors = self.embedder.embed_texts([query])
                if not vectors:
                    logger.warning(
                        "Empty query after preprocessing. Falling back to BM25."
                    )
                    return execute_query_bm25(query)
                query_vector = vectors[0]

                def build_vector_query(nct_filter: Dict) -> Dict:
                    return {
                        "script_score": {
                            "query": {"bool": {"filter": nct_filter}},
                            "script": {
                                "source": """
                                double vectorScore = (cosineSimilarity(params.query_vector, 'criterion_vector') + 1.0) / 2.0;
                                if (vectorScore < params.vector_score_threshold) {
                                    return 0;
                                }
                                return vectorScore;
                            """,
                                "params": {
                                    "query_vector": query_vector,
                                    "vector_score_threshold": (
                                        self.second_level_vector_score_threshold
                                    ),
                                },
                            },
                        }
                    }

                try:
                    hits = self._search_criteria_batched(
                        nct_ids, build_vector_query, "ES criteria search"
                    )
                    logger.info(
                        "[%s] Retrieved %s documents for query: '%s'",
                        self.search_mode,
                        len(hits),
                        query,
                    )
                    return query, hits
                except Exception:
                    logger.exception(
                        "Second-level search failed for query: %s", query
                    )
                    return query, []

            # Hybrid mode (default)
            if self.embedder is None:
                logger.warning(
                    "Hybrid mode selected but embedder is None. Falling back to BM25."
                )
                return execute_query_bm25(query)

            vectors = self.embedder.embed_texts([query])
            if not vectors:
                logger.warning(
                    "Empty query after preprocessing. Falling back to BM25."
                )
                return execute_query_bm25(query)
            query_vector = vectors[0]

            def build_hybrid_query(nct_filter: Dict) -> Dict:
                return {
                    "script_score": {
                        "query": {
                            "bool": {
                                "should": [
                                    {
                                        "multi_match": {
                                            "query": query,
                                            "fields": fields_to_search,
                                            "type": "best_fields",
                                            "operator": "and",
                                        }
                                    },
                                    {
                                        "multi_match": {
                                            "query": query,
                                            "fields": fields_to_search,
                                            "type": "phrase",
                                        }
                                    },
                                    {
                                        "multi_match": {
                                            "query": query,
                                            "fields": fields_to_search,
                                            "type": "best_fields",
                                            "operator": "or",
                                        }
                                    },
                                ],
                                "minimum_should_match": 1,
                                "filter": nct_filter,
                            }
                        },
                        "script": {
                            "source": """
                                double alpha = 0.5;
                                double beta = 0.5;
                                double textScore = _score;
                                double vectorScore = (cosineSimilarity(params.query_vector, 'criterion_vector') + 1.0) / 2.0;
                                if (vectorScore < params.vector_score_threshold) {
                                    return 0;
                                }
                                return alpha * textScore + beta * vectorScore;
                            """,
                            "params": {
                                "query_vector": query_vector,
                                "vector_score_threshold": (
                                    self.second_level_vector_score_threshold
                                ),
                            },
                        },
                    }
                }

            try:
                hits = self._search_criteria_batched(
                    nct_ids, build_hybrid_query, "ES criteria search"
                )
                logger.info(
                    "[%s] Retrieved %s documents for query: '%s'",
                    self.search_mode,
                    len(hits),
                    query,
                )
                return query, hits
            except Exception:
                logger.exception("Second-level search failed for query: %s", query)
                return query, []

        def execute_query_bm25(query):
            fields_to_search = (
                ["criterion", "entities.synonyms"]
                if self.bio_med_ner is not None
                else ["criterion"]
            )

            def build_bm25_fallback(nct_filter: Dict) -> Dict:
                return {
                    "bool": {
                        "should": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": fields_to_search,
                                    "type": "best_fields",
                                    "operator": "and",
                                }
                            },
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": fields_to_search,
                                    "type": "phrase",
                                }
                            },
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": fields_to_search,
                                    "type": "best_fields",
                                    "operator": "or",
                                }
                            },
                        ],
                        "minimum_should_match": 1,
                        "filter": nct_filter,
                    }
                }

            try:
                hits = self._search_criteria_batched(
                    nct_ids, build_bm25_fallback, "ES criteria bm25 search"
                )
                logger.info(
                    "[bm25] Retrieved %s documents for query: '%s'", len(hits), query
                )
                return query, hits
            except Exception:
                logger.exception("BM25 search failed for query: %s", query)
                return query, []

        with ThreadPoolExecutor(max_workers=min(8, len(queries))) as executor:
            future_to_query = {
                executor.submit(execute_query, query): query for query in queries
            }
            for future in as_completed(future_to_query):
                query, hits = future.result()
                query_to_hits[query] = hits
        return query_to_hits

    def rerank_criteria(self, queries: List[str], criteria: List[Dict]) -> List[Dict]:
        if self.llm_reranker is None:
            logger.warning("LLM reranker not available, using ES scores only")
            return self.score_criteria_without_llm(criteria)

        pairs = [
            (criterion["query"], criterion["_source"]["criterion"])
            for criterion in criteria
        ]
        llm_scores = self.llm_reranker.rank_pairs(pairs)
        llm_scores = [
            score.get("llm_score", 0.0) if isinstance(score, dict) else float(score)
            for score in llm_scores
        ]
        if len(llm_scores) != len(pairs):
            logger.error("Mismatch between LLM scores and pairs!")
            raise ValueError("Mismatch between LLM scores and pairs!")
        for i, criterion in enumerate(criteria):
            llm_score = llm_scores[i]
            eligibility_type = criterion["_source"].get("eligibility_type", "").lower()
            if eligibility_type == "inclusion criteria":
                llm_score *= self.inclusion_weight
            elif eligibility_type == "exclusion criteria":
                llm_score *= self.exclusion_weight
            criterion["llm_score"] = llm_score
        return criteria

    def score_criteria_without_llm(self, criteria: List[Dict]) -> List[Dict]:
        if not criteria:
            return criteria
        max_es = max((c.get("_score", 0.0) for c in criteria), default=1.0) or 1.0
        for criterion in criteria:
            base = float(criterion.get("_score", 0.0)) / max_es
            eligibility_type = criterion["_source"].get("eligibility_type", "").lower()
            if eligibility_type == "inclusion criteria":
                base *= self.inclusion_weight
            elif eligibility_type == "exclusion criteria":
                base *= self.exclusion_weight
            criterion["llm_score"] = base
        return criteria

    def aggregate_to_trials(
        self, criteria: List[Dict], threshold: float = 0.5, method: str = "weighted"
    ) -> List[Dict]:
        trial_scores = defaultdict(list)
        for criterion in criteria:
            nct_id = criterion["_source"]["nct_id"]
            score = criterion["llm_score"]
            if score >= threshold:
                trial_scores[nct_id].append(score)
        aggregated_scores = {}
        for nct_id, scores in trial_scores.items():
            count = len(scores)
            total = sum(scores)
            if count == 0:
                continue
            if method == "avg":
                agg_score = total / count
            elif method == "sqrt":
                agg_score = total / math.sqrt(count)
            elif method == "log":
                agg_score = total / math.log(count + 1)
            elif method == "weighted":
                agg_score = 0.7 * (total / math.sqrt(count)) + 0.3 * max(scores)
            else:
                raise ValueError(f"Unsupported aggregation method: {method}")
            aggregated_scores[nct_id] = agg_score
        sorted_trials = [
            {"nct_id": nct_id, "score": score}
            for nct_id, score in sorted(
                aggregated_scores.items(), key=lambda x: x[1], reverse=True
            )
        ]
        return sorted_trials

    def retrieve_and_rank(
        self,
        queries: List[str],
        nct_ids: List[str],
        top_n: int,
        use_reranker: bool = True,
        save_path: Optional[str] = None,
        missing_criteria_nct_ids_out: Optional[str] = None,
    ) -> List[Dict]:
        # Cap queries to prevent memory/performance issues
        max_queries = 150  # Reasonable limit for second-level search
        if len(queries) > max_queries:
            logger.warning(
                f"Capping queries from {len(queries)} to {max_queries} for second-level search"
            )
            queries = queries[:max_queries]

        all_criteria = []
        if self.search_mode == "all_rerank":
            base_criteria = self.retrieve_all_criteria(nct_ids)
            for query in queries:
                for hit in base_criteria:
                    criterion = dict(hit)
                    criterion["query"] = query
                    all_criteria.append(criterion)
            logger.info(
                "[%s] Constructed %s query-criterion pairs from %s queries and %s criteria documents",
                self.search_mode,
                len(all_criteria),
                len(queries),
                len(base_criteria),
            )
        else:
            query_to_hits = self.retrieve_criteria(nct_ids, queries)
            for query, hits in query_to_hits.items():
                for hit in hits:
                    hit["query"] = query
                    all_criteria.append(hit)

        if missing_criteria_nct_ids_out:
            missing = nct_ids_without_criterion_hits(nct_ids, all_criteria)
            write_text_file(missing, missing_criteria_nct_ids_out)
            logger.info(
                "Wrote %d NCT id(s) with no criterion documents in this retrieval pass to %s",
                len(missing),
                missing_criteria_nct_ids_out,
            )

        # Check if reranker is available before trying to use it
        if use_reranker and self.llm_reranker is not None:
            ranked_criteria = self.rerank_criteria(queries, all_criteria)
        else:
            if use_reranker and self.llm_reranker is None:
                logger.info(
                    "Reranking requested but LLM reranker not available; using ES scores for aggregation."
                )
            else:
                logger.info(
                    "Second-level reranking disabled; using ES scores for aggregation."
                )
            ranked_criteria = self.score_criteria_without_llm(all_criteria)

        sorted_trials = self.aggregate_to_trials(
            ranked_criteria, threshold=self.second_level_aggregate_score_threshold
        )
        top_trials = sorted_trials[:top_n]
        logger.info(f"Top {top_n} trials retrieved: {top_trials}")
        if save_path:
            write_text_file([trial["nct_id"] for trial in top_trials], save_path)
            logger.info(f"Top trials saved to {save_path}")
        return top_trials

