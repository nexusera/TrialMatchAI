from datetime import datetime
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union

from dateutil import parser as date_parser
from Matcher.models.embedding.text_embedder import TextEmbedder
from Matcher.utils.logging_config import setup_logging
from Matcher.utils.retry import with_retries

from elasticsearch import BadRequestError, Elasticsearch
from elasticsearch.helpers import scan

logger = setup_logging(__name__)

# Indexed trials often carry maximum_age=0 (or negative) as a placeholder when the
# source registry has no upper bound. Treat as unbounded for ES and diagnostics.
# Genuine pediatric upper limits are typically >= 1 year.

# Helper that returns 0.0 when the doc is missing a vector field instead of
# crashing with a Painless runtime error.
_SAFE_COSINE = (
    "double _safe(def qv, String f, def doc) {"
    "  if (!doc.containsKey(f) || doc[f].size() == 0) return 0.0;"
    "  return cosineSimilarity(qv, f);"
    "}"
)

_VECTOR_ONLY_SCRIPT = _SAFE_COSINE + """
    double maxConditionVectorScore = 0.0;
    double maxTitleVectorScore = 0.0;
    double maxSummaryVectorScore = 0.0;
    double maxEligibilityVectorScore = 0.0;
    double totalOtherConditionScore = 0.0;
    for (int i = 0; i < params.query_vectors.length; ++i) {
        maxConditionVectorScore  = Math.max(maxConditionVectorScore,  _safe(params.query_vectors[i], 'condition_vector', doc));
        maxTitleVectorScore      = Math.max(maxTitleVectorScore,      _safe(params.query_vectors[i], 'brief_title_vector', doc));
        maxSummaryVectorScore    = Math.max(maxSummaryVectorScore,    _safe(params.query_vectors[i], 'brief_summary_vector', doc));
        maxEligibilityVectorScore= Math.max(maxEligibilityVectorScore,_safe(params.query_vectors[i], 'eligibility_criteria_vector', doc));
    }
    int otherConditionCount = params.other_condition_vectors.length;
    for (int i = 0; i < otherConditionCount; ++i) {
        totalOtherConditionScore += _safe(params.other_condition_vectors[i], 'condition_vector', doc);
        totalOtherConditionScore += _safe(params.other_condition_vectors[i], 'brief_title_vector', doc);
        totalOtherConditionScore += _safe(params.other_condition_vectors[i], 'eligibility_criteria_vector', doc);
        totalOtherConditionScore += _safe(params.other_condition_vectors[i], 'brief_summary_vector', doc);
    }
    if (otherConditionCount > 0) {
        totalOtherConditionScore /= (otherConditionCount * 4);
    }
    double normalizedConditionScore      = (maxConditionVectorScore + 1.0) / 2.0;
    double normalizedTitleScore          = (maxTitleVectorScore + 1.0) / 2.0;
    double normalizedSummaryScore        = (maxSummaryVectorScore + 1.0) / 2.0;
    double normalizedEligibilityScore    = (maxEligibilityVectorScore + 1.0) / 2.0;
    double normalizedOtherConditionScore = (totalOtherConditionScore + 1.0) / 2.0;
    double combinedVectorScore = (
        0.3 * normalizedConditionScore +
        0.1 * normalizedTitleScore +
        0.1 * normalizedSummaryScore +
        0.2 * normalizedOtherConditionScore +
        0.3 * normalizedEligibilityScore
    );
    if (combinedVectorScore < params.vector_score_threshold) {
        return 0;
    }
    return combinedVectorScore;
"""

_HYBRID_SCRIPT = _SAFE_COSINE + """
    double alpha = 0.5;
    double beta = 0.5;
    double textScore = _score;
    double maxTextScore = params.max_text_score;
    double normalizedTextScore = (maxTextScore == 0) ? 0 : textScore / maxTextScore;
    double maxConditionVectorScore = 0.0;
    double maxTitleVectorScore = 0.0;
    double maxSummaryVectorScore = 0.0;
    double maxEligibilityVectorScore = 0.0;
    double totalOtherConditionScore = 0.0;
    for (int i = 0; i < params.query_vectors.length; ++i) {
        maxConditionVectorScore  = Math.max(maxConditionVectorScore,  _safe(params.query_vectors[i], 'condition_vector', doc));
        maxTitleVectorScore      = Math.max(maxTitleVectorScore,      _safe(params.query_vectors[i], 'brief_title_vector', doc));
        maxSummaryVectorScore    = Math.max(maxSummaryVectorScore,    _safe(params.query_vectors[i], 'brief_summary_vector', doc));
        maxEligibilityVectorScore= Math.max(maxEligibilityVectorScore,_safe(params.query_vectors[i], 'eligibility_criteria_vector', doc));
    }
    int otherConditionCount = params.other_condition_vectors.length;
    for (int i = 0; i < otherConditionCount; ++i) {
        totalOtherConditionScore += _safe(params.other_condition_vectors[i], 'condition_vector', doc);
        totalOtherConditionScore += _safe(params.other_condition_vectors[i], 'brief_title_vector', doc);
        totalOtherConditionScore += _safe(params.other_condition_vectors[i], 'eligibility_criteria_vector', doc);
        totalOtherConditionScore += _safe(params.other_condition_vectors[i], 'brief_summary_vector', doc);
    }
    if (otherConditionCount > 0) {
        totalOtherConditionScore /= (otherConditionCount * 4);
    }
    double normalizedConditionScore      = (maxConditionVectorScore + 1.0) / 2.0;
    double normalizedTitleScore          = (maxTitleVectorScore + 1.0) / 2.0;
    double normalizedSummaryScore        = (maxSummaryVectorScore + 1.0) / 2.0;
    double normalizedEligibilityScore    = (maxEligibilityVectorScore + 1.0) / 2.0;
    double normalizedOtherConditionScore = (totalOtherConditionScore + 1.0) / 2.0;
    double combinedVectorScore = (
        0.3 * normalizedConditionScore +
        0.1 * normalizedTitleScore +
        0.1 * normalizedSummaryScore +
        0.2 * normalizedOtherConditionScore +
        0.3 * normalizedEligibilityScore
    );
    if (combinedVectorScore < params.vector_score_threshold) {
        return 0;
    }
    return alpha * normalizedTextScore + beta * combinedVectorScore;
"""

# Hybrid/vector scripts return exactly 0 when combinedVectorScore < vector_score_threshold.
# Without min_score, those docs still appear in hits (sorted last). This floor drops them so
# --max-trials-first-level is filled with threshold-passing trials only (when enough exist).
_FIRST_LEVEL_MIN_SCORE_AFTER_VECTOR_THRESHOLD = 1e-9


class EligibilityFilterBuild(NamedTuple):
    filters: List[dict]
    gender_terms: List[str]
    age_filter_enabled: bool
    patient_age_for_range: int
    status_filter_enabled: bool
    preselected_enabled: bool


def build_eligibility_filters(
    age: Union[int, None, str],
    sex: str,
    overall_status: Optional[str],
    pre_selected_nct_ids: Optional[List[str]],
) -> EligibilityFilterBuild:
    """Build the same bool.filter clauses as first-level trial search (age/sex/status/preselect)."""
    sex_u = (sex or "all").upper()
    gender_terms = {
        "MALE": ["MALE", "Male", "male", "M", "All", "all", "ALL"],
        "FEMALE": ["FEMALE", "Female", "female", "F", "All", "all", "ALL"],
        "ALL": [
            "All",
            "all",
            "ALL",
            "Both",
            "both",
            "BOTH",
            "FEMALE",
            "Female",
            "female",
            "F",
            "MALE",
            "Male",
            "male",
            "M",
        ],
    }.get(sex_u, ["All"])
    filters: List[dict] = []
    age_filter_enabled = age not in [None, "all", "ALL", "All"]
    patient_age_int = 0
    if age_filter_enabled:
        try:
            patient_age_int = int(age)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            patient_age_int = 0
        filters.append(
            {
                "bool": {
                    "must": [
                        {
                            "bool": {
                                "should": [
                                    {"range": {"minimum_age": {"lte": patient_age_int}}},
                                    {
                                        "bool": {
                                            "must_not": {
                                                "exists": {"field": "minimum_age"}
                                            }
                                        }
                                    },
                                ]
                            }
                        },
                        {
                            "bool": {
                                "should": [
                                    {"range": {"maximum_age": {"gte": patient_age_int}}},
                                    {"range": {"maximum_age": {"lte": 0}}},
                                    {
                                        "bool": {
                                            "must_not": {
                                                "exists": {"field": "maximum_age"}
                                            }
                                        }
                                    },
                                ]
                            }
                        },
                    ]
                }
            }
        )
    status_filter_enabled = bool(overall_status and overall_status.lower() != "all")
    if status_filter_enabled:
        filters.append({"match": {"overall_status": overall_status}})
    preselected_enabled = bool(pre_selected_nct_ids)
    if preselected_enabled:
        filters.append({"terms": {"nct_id": pre_selected_nct_ids}})
    if gender_terms:
        filters.append(
            {
                "bool": {
                    "should": [
                        {"terms": {"gender": gender_terms}},
                        {"bool": {"must_not": {"exists": {"field": "gender"}}}},
                    ]
                }
            }
        )
    return EligibilityFilterBuild(
        filters=filters,
        gender_terms=gender_terms,
        age_filter_enabled=age_filter_enabled,
        patient_age_for_range=patient_age_int,
        status_filter_enabled=status_filter_enabled,
        preselected_enabled=preselected_enabled,
    )


def _diagnose_trial_filter_miss(
    src: Dict[str, Any],
    fb: EligibilityFilterBuild,
    overall_status_query: Optional[str],
    pre_selected_ids: Optional[List[str]],
) -> List[str]:
    """Best-effort human-readable reasons a trial may fail first-level ES filters."""
    reasons: List[str] = []
    if fb.age_filter_enabled:
        pa = fb.patient_age_for_range
        mn = src.get("minimum_age")
        if mn is not None and mn != "":
            try:
                mnf = float(mn)
                if mnf > pa:
                    reasons.append(
                        f"年龄下限: 试验 minimum_age={mnf:g} > 用于 filter 的患者年龄 {pa}"
                    )
            except (TypeError, ValueError):
                reasons.append(f"minimum_age 无法解析为数字: {mn!r}")
        mx = src.get("maximum_age")
        if mx is not None and mx != "":
            try:
                mxf = float(mx)
                if mxf > 0 and mxf < pa:
                    reasons.append(
                        f"年龄上限: 试验 maximum_age={mxf:g} < 用于 filter 的患者年龄 {pa}"
                    )
            except (TypeError, ValueError):
                reasons.append(f"maximum_age 无法解析为数字: {mx!r}")

    gval = src.get("gender")
    if gval is not None and str(gval).strip() != "":
        if str(gval) not in fb.gender_terms:
            reasons.append(
                f"性别: 试验 gender={gval!r} 不在当前患者允许集合（与 ES terms 一致）"
            )

    if fb.status_filter_enabled and overall_status_query:
        ov = src.get("overall_status")
        q = str(overall_status_query).strip().lower()
        ovs = (str(ov).strip().lower() if ov is not None else "")
        if not ovs:
            reasons.append("招募状态: 试验缺少 overall_status，难以通过状态 filter")
        elif q not in ovs and ovs not in q:
            reasons.append(
                f"招募状态: 试验 overall_status={ov!r} 与 filter 关键词 {overall_status_query!r} "
                "（ES match）可能不命中"
            )

    if fb.preselected_enabled and pre_selected_ids:
        tid = str(src.get("nct_id") or "")
        if tid and tid not in set(pre_selected_ids):
            reasons.append("不在 pre_selected_nct_ids 白名单中")

    if not reasons:
        reasons.append(
            "未匹配到明确的年龄/性别/状态/白名单原因（可能与 ES 字段类型、analyzer 或脚本分有关）"
        )
    return reasons


def explain_first_level_filter_misses(
    es_client: Elasticsearch,
    index_name: str,
    *,
    fb: EligibilityFilterBuild,
    retrieved_nct_ids: List[str],
    max_trials_first_level: int,
    sex: str,
    overall_status: Optional[str],
    pre_selected_nct_ids: Optional[List[str]],
    patient_age_raw: Any,
    vector_score_threshold: float,
) -> Dict[str, Any]:
    """Compare index vs filter-only pass set vs hybrid top-K; write-oriented payload."""
    retrieved: Set[str] = {str(x) for x in retrieved_nct_ids if x}

    # scan()'s `query` is copied into client.search(**kwargs). A bare {"match_all": {}}
    # would become invalid kwargs (match_all=...). Pass Query DSL under "query".
    _src_fields = ["nct_id", "minimum_age", "maximum_age", "gender", "overall_status"]
    all_sources: Dict[str, Dict[str, Any]] = {}
    for hit in scan(
        es_client,
        index=index_name,
        query={"query": {"match_all": {}}},
        source_includes=_src_fields,
        size=500,
    ):
        src = hit.get("_source") or {}
        nid = src.get("nct_id")
        if nid:
            all_sources[str(nid)] = src

    filter_pass: Set[str] = set()
    inner_q: Dict[str, Any] = (
        {"bool": {"filter": fb.filters}} if fb.filters else {"match_all": {}}
    )
    for hit in scan(
        es_client,
        index=index_name,
        query={"query": inner_q},
        source_includes=["nct_id"],
        size=500,
    ):
        src = hit.get("_source") or {}
        nid = src.get("nct_id")
        if nid:
            filter_pass.add(str(nid))

    failed_filter_full: List[Dict[str, Any]] = []
    for nid, src in sorted(all_sources.items()):
        if nid not in filter_pass:
            failed_filter_full.append(
                {
                    "nct_id": nid,
                    "reasons": _diagnose_trial_filter_miss(
                        src, fb, overall_status, pre_selected_nct_ids
                    ),
                }
            )
    max_failed_list = 5000
    failed_truncated = len(failed_filter_full) > max_failed_list
    failed_filter = failed_filter_full[:max_failed_list]

    passed_not_in_topk_sorted = sorted(filter_pass - retrieved)
    max_list = 2000
    passed_slice = passed_not_in_topk_sorted[:max_list]
    truncated = len(passed_not_in_topk_sorted) > max_list
    in_topk = sorted(retrieved)

    return {
        "index": index_name,
        "patient_age_raw": patient_age_raw,
        "patient_sex": sex,
        "overall_status_filter": overall_status,
        "vector_score_threshold": vector_score_threshold,
        "max_trials_first_level": max_trials_first_level,
        "counts": {
            "docs_in_index": len(all_sources),
            "passed_eligibility_filters": len(filter_pass),
            "retrieved_first_level": len(retrieved),
            "failed_eligibility_filters": len(failed_filter_full),
            "passed_filters_but_not_in_topk": len(passed_not_in_topk_sorted),
        },
        "filter_flags": {
            "age_filter_enabled": fb.age_filter_enabled,
            "patient_age_used_in_filter": fb.patient_age_for_range,
            "status_filter_enabled": fb.status_filter_enabled,
            "preselected_enabled": fb.preselected_enabled,
        },
        "note": (
            "passed_filters_but_not_in_topk: 已通过年龄/性别/状态等 filter，"
            "但未进入一级 hybrid/bm25 返回的前 max_trials_first_level 条（排序/截断）。"
        ),
        "failed_eligibility_filters_truncated": failed_truncated,
        "failed_eligibility_filters_omitted": (
            len(failed_filter_full) - max_failed_list if failed_truncated else 0
        ),
        "failed_eligibility_filters": failed_filter,
        "passed_filters_but_not_in_topk_truncated": truncated,
        "passed_filters_but_not_in_topk_omitted": (
            len(passed_not_in_topk_sorted) - max_list if truncated else 0
        ),
        "passed_filters_but_not_in_topk": [
            {
                "nct_id": nid,
                "reasons": [
                    f"已通过 filter，但未进入一级检索前 {max_trials_first_level} 名（hybrid 综合分排序）"
                ],
            }
            for nid in passed_slice
        ],
        "retrieved_nct_ids_in_order": in_topk,
    }


class ClinicalTrialSearch:
    def __init__(
        self,
        es_client: Elasticsearch,
        embedder: Optional[TextEmbedder],
        index_name: str,
        bio_med_ner,
    ):
        self.es_client = es_client
        self.embedder = embedder
        self.index_name = index_name
        self.bio_med_ner = bio_med_ner

    def get_synonyms(self, condition: str) -> List[str]:
        if not self.bio_med_ner:
            logger.info("BioMedNER disabled; skipping synonyms extraction.")
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
        except Exception as e:
            logger.error(f"BioMedNER synonym extraction failed for '{condition}': {e}")
        return []

    def parse_age_input(self, age_input: Union[int, str]) -> Optional[int]:
        if isinstance(age_input, int):
            return age_input
        elif isinstance(age_input, str):
            age_input = age_input.strip().lower()
            age_keywords = ["year", "years", "yr", "yrs"]
            try:
                for keyword in age_keywords:
                    if age_input.endswith(keyword):
                        age_str = age_input.replace(keyword, "").strip()
                        return int(age_str)
                return int(age_input)
            except ValueError:
                pass
            try:
                dob = date_parser.parse(age_input, fuzzy=True)
                today = datetime.today()
                age = (
                    today.year
                    - dob.year
                    - ((today.month, today.day) < (dob.month, dob.day))
                )
                if age < 0:
                    raise ValueError("Invalid date of birth.")
                return age
            except (ValueError, OverflowError, date_parser.ParserError):
                pass
        return None

    def get_max_text_score(self, synonyms: List[str]) -> float:
        should_clauses = [
            {
                "multi_match": {
                    "query": syn,
                    "fields": [
                        "condition^6",
                        "eligibility_criteria^4",
                        "brief_title^3",
                        "brief_summary^2",
                        "detailed_description^1.5",
                        "official_title",
                    ],
                    "type": "best_fields",
                    "operator": "and",
                }
            }
            for syn in synonyms
        ]
        try:
            response = with_retries(
                lambda: self.es_client.search(
                    index=self.index_name,
                    body={
                        "size": 1,
                        "query": {
                            "bool": {"should": should_clauses, "minimum_should_match": 0}
                        },
                        "track_total_hits": False,
                        "_source": False,
                    },
                ),
                logger=logger,
                action="ES max_text_score search",
            )
            max_score = response["hits"]["max_score"]
            return max_score if max_score else 1.0
        except Exception:
            logger.exception("Failed to compute max text score; defaulting to 1.0")
            return 1.0

    def create_query(
        self,
        synonyms: List[str],
        embeddings: Dict[str, List[float]],
        age: Optional[int],
        sex: str,
        overall_status: Optional[str],
        max_text_score: float,
        vector_score_threshold: float = 0.5,
        pre_selected_nct_ids: Optional[List[str]] = None,
        other_conditions: Optional[List[str]] = None,
        search_mode: str = "hybrid",
    ) -> Dict:
        fb = build_eligibility_filters(
            age, sex, overall_status, pre_selected_nct_ids
        )
        filters = fb.filters

        # Cap conditions to prevent too many ES clauses (each condition creates 2 clauses)
        # ES default maxClauseCount is 1024, leaving room for filters and other clauses
        max_conditions_per_query = 800  # Conservative limit
        all_conditions = synonyms + (other_conditions or [])
        if len(all_conditions) > max_conditions_per_query:
            logger.warning(
                f"Capping search conditions from {len(all_conditions)} to {max_conditions_per_query} to avoid ES clause limit"
            )
            # Prioritize synonyms over other_conditions
            capped_synonyms = synonyms[:max_conditions_per_query]
            remaining_slots = max_conditions_per_query - len(capped_synonyms)
            capped_other = (
                (other_conditions or [])[:remaining_slots]
                if remaining_slots > 0
                else []
            )
            synonyms = capped_synonyms
            other_conditions = capped_other

        should_clauses = []
        for condition in synonyms + (other_conditions or []):
            if condition:
                for match_type in ["best_fields", "phrase"]:
                    multi_match = {
                        "query": condition,
                        "fields": [
                            "condition^6",
                            "eligibility_criteria^4",
                            "brief_title^3",
                            "brief_summary^2",
                            "detailed_description^1.5",
                            "official_title",
                        ],
                        "type": match_type,
                    }
                    if match_type == "best_fields":
                        multi_match["operator"] = "and"
                    should_clauses.append({"multi_match": multi_match})

        logger.info(
            f"Created query with {len(should_clauses)} should clauses for {len(synonyms)} synonyms and {len(other_conditions or [])} other conditions"
        )

        search_mode = (search_mode or "hybrid").lower()
        if search_mode == "bm25":
            return {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 0,
                    "filter": filters,
                }
            }

        # Prepare vectors for vector/hybrid
        query_vectors = [
            embeddings[term] for term in synonyms if term in embeddings and term
        ]
        other_vectors = [
            embeddings[term]
            for term in (other_conditions or [])
            if term in embeddings and term
        ]

        if search_mode == "vector":
            return {
                "script_score": {
                    "query": {
                        "bool": {
                            "filter": filters,
                        }
                    },
                    "script": {
                        "source": _VECTOR_ONLY_SCRIPT,
                        "params": {
                            "query_vectors": query_vectors,
                            "other_condition_vectors": other_vectors,
                            "vector_score_threshold": vector_score_threshold,
                        },
                    },
                }
            }

        # Hybrid (default)
        return {
            "script_score": {
                "query": {
                    "bool": {
                        "should": should_clauses,
                        "minimum_should_match": 0,
                        "filter": filters,
                    }
                },
                "script": {
                    "source": _HYBRID_SCRIPT,
                    "params": {
                        "query_vectors": query_vectors,
                        "other_condition_vectors": other_vectors,
                        "max_text_score": max_text_score,
                        "vector_score_threshold": vector_score_threshold,
                    },
                },
            }
        }

    def search_trials(
        self,
        condition: str,
        age_input: Union[int, str],
        sex: str,
        overall_status: Optional[str] = None,
        size: int = 10,
        pre_selected_nct_ids: Optional[List[str]] = None,
        synonyms: Optional[List[str]] = None,
        other_conditions: Optional[List[str]] = None,
        vector_score_threshold: float = 0.0,
        search_mode: str = "hybrid",
    ) -> Tuple[List[Dict], List[float]]:
        if age_input not in ["all", "ALL", "All"]:
            age = self.parse_age_input(age_input)
            if age is None:
                raise ValueError("Could not parse age input.")
        else:
            age = None
        primary_synonyms = _clean_terms([condition] + (synonyms or []))
        other_conditions = _clean_terms(other_conditions or [])
        all_terms = primary_synonyms + other_conditions

        mode = (search_mode or "hybrid").lower()
        embeddings: Dict[str, List[float]] = {}
        if mode in {"vector", "hybrid"} and self.embedder is not None:
            vectors = self.embedder.embed_texts(all_terms)
            embeddings = dict(zip(all_terms, vectors))
            if not embeddings:
                logger.warning(
                    "No valid terms to embed for vector search. Falling back to BM25 only."
                )
                mode = "bm25"
        elif mode in {"vector", "hybrid"} and self.embedder is None:
            logger.warning(
                "Vector/hybrid mode selected but embedder is None. Falling back to BM25 only."
            )
            mode = "bm25"

        max_text_score = (
            1.0 if mode == "vector" else self.get_max_text_score(primary_synonyms)
        )
        query = self.create_query(
            primary_synonyms,
            embeddings,
            age,
            sex,
            overall_status,
            max_text_score,
            vector_score_threshold,
            pre_selected_nct_ids,
            other_conditions,
            search_mode=mode,
        )
        try:
            search_body: Dict[str, Any] = {"size": size, "query": query}
            if mode in {"hybrid", "vector"}:
                search_body["min_score"] = _FIRST_LEVEL_MIN_SCORE_AFTER_VECTOR_THRESHOLD
            response = with_retries(
                lambda: self.es_client.search(
                    index=self.index_name, body=search_body
                ),
                logger=logger,
                action="ES trial search",
            )
            hits = response["hits"]["hits"]
            trials = [hit["_source"] for hit in hits]
            scores = [hit["_score"] for hit in hits]
        except BadRequestError as exc:
            # Typical cause: dense_vector runtime mismatch (e.g., custom index dims differ
            # from current embedder output). Fall back to BM25 so pipeline can continue.
            if mode in {"vector", "hybrid"}:
                logger.warning(
                    "Vector/hybrid ES query failed (%s). Falling back to BM25 for this request.",
                    exc,
                )
                bm25_query = self.create_query(
                    primary_synonyms,
                    embeddings={},
                    age=age,
                    sex=sex,
                    overall_status=overall_status,
                    max_text_score=1.0,
                    vector_score_threshold=vector_score_threshold,
                    pre_selected_nct_ids=pre_selected_nct_ids,
                    other_conditions=other_conditions,
                    search_mode="bm25",
                )
                response = with_retries(
                    lambda: self.es_client.search(
                        index=self.index_name, body={"size": size, "query": bm25_query}
                    ),
                    logger=logger,
                    action="ES trial search (bm25 fallback)",
                )
                hits = response["hits"]["hits"]
                trials = [hit["_source"] for hit in hits]
                scores = [hit["_score"] for hit in hits]
            else:
                logger.exception("Search failed; returning empty results.")
                return [], []
        except Exception:
            logger.exception("Search failed; returning empty results.")
            return [], []
        trials_with_scores = sorted(
            zip(trials, scores), key=lambda x: x[1], reverse=True
        )
        top_x_percent_index = int(len(trials_with_scores) * 1.0)
        sliced = trials_with_scores[:top_x_percent_index]
        trials = [trial for trial, score in sliced]
        scores = [score for trial, score in sliced]
        logger.info(
            f"[{mode}] Found {len(trials)} trials matching the search criteria."
        )
        return trials, scores


def _clean_terms(terms: List[str]) -> List[str]:
    return [term.strip() for term in terms if term and term.strip()]

