from __future__ import annotations

import argparse
import importlib
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from Parser.biomedner_engine import BioMedNER

from elasticsearch import Elasticsearch

from Matcher.config.config_loader import load_config
from Matcher.models.embedding.text_embedder import TextEmbedder, TextEmbedderConfig
from Matcher.models.llm.llm_loader import load_model_and_tokenizer
from Matcher.models.llm.llm_reranker import LLMReranker
from Matcher.pipeline.cot_reasoning import BatchTrialProcessor
from Matcher.pipeline.cot_reasoning_vllm import BatchTrialProcessorVLLM
from Matcher.pipeline.phenopacket_processor import process_phenopacket
from Matcher.pipeline.trial_ranker import (
    load_trial_data,
    rank_trials,
    save_ranked_trials,
)
from Matcher.pipeline.trial_search.first_level_search import (
    ClinicalTrialSearch,
    build_eligibility_filters,
    explain_first_level_filter_misses,
)
from Matcher.pipeline.trial_search.second_level_search import SecondStageRetriever
from Matcher.services.biomedner_service import initialize_biomedner_services
from Matcher.services.elasticsearch_service import ensure_elasticsearch
from Matcher.utils.file_utils import (
    create_directory,
    read_json_file,
    read_text_file,
    write_json_file,
    write_text_file,
)
from Matcher.schemas.phenopacket import Keywords, Phenopacket
from Matcher.utils.logging_config import reset_request_id, set_request_id, setup_logging
from Matcher.utils.temporal_utils import infer_patient_age_years_from_phenopacket
from Matcher.utils.timing import log_timing

logger = setup_logging(__name__)


def _resolve_vllm_loader():
    """Resolve vLLM loader across minor API name differences."""
    mod = importlib.import_module("Matcher.models.llm.vllm_loader")
    for name in ("load_vllm_engine", "build_vllm_engine", "load_engine"):
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    available = [n for n in dir(mod) if "vllm" in n.lower() or "engine" in n.lower()]
    raise ImportError(
        "Cannot find a vLLM loader function in Matcher.models.llm.vllm_loader. "
        f"Tried: load_vllm_engine/build_vllm_engine/load_engine. Available symbols: {available}"
    )


def load_all_trial_ids_from_folder(trials_json_folder: str) -> List[str]:
    """Load candidate trial IDs from local trial JSON files."""
    folder = Path(trials_json_folder)
    if not folder.exists():
        logger.error("Trials JSON folder not found: %s", folder)
        return []
    trial_ids = sorted({p.stem for p in folder.iterdir() if p.suffix == ".json"})
    logger.info(
        "Loaded %d candidate trial IDs from %s for second-level retrieval.",
        len(trial_ids),
        folder,
    )
    return trial_ids


# ── Pipeline stages ─────────────────────────────────────────────────────


def run_first_level_search(
    keywords: Dict,
    output_folder: str,
    patient_info: Dict,
    bio_med_ner,
    embedder: TextEmbedder,
    config: Dict,
    es_client: Elasticsearch,
    *,
    explain_filter_misses: bool = False,
) -> Optional[Tuple]:
    main_conditions = keywords.get("main_conditions", [])
    other_conditions = keywords.get("other_conditions", [])
    expanded_sentences = keywords.get("expanded_sentences", [])

    if not main_conditions:
        logger.error("No main_conditions found in keywords.")
        return None

    condition = main_conditions[0]
    age_input = patient_info.get("age", "all")
    sex = patient_info.get("gender", "all")
    overall_status = "All"

    index_name = config["elasticsearch"]["index_trials"]
    cts = ClinicalTrialSearch(es_client, embedder, index_name, bio_med_ner)

    synonyms = cts.get_synonyms(condition.lower().strip())
    main_conditions.extend(synonyms[:5])

    if age_input not in ["all", "ALL", "All"]:
        _parsed_age = cts.parse_age_input(age_input)
        if _parsed_age is None:
            logger.error("Could not parse patient age for first-level search: %r", age_input)
            return None
        age_for_query = _parsed_age
    else:
        # Unknown age must not become 0 — that enables minimum_age filters and drops adult trials.
        age_for_query = None

    search_size = config["search"].get("max_trials_first_level", 300)
    trials, scores = cts.search_trials(
        condition=condition,
        age_input=age_input,
        sex=sex,
        overall_status=overall_status,
        size=search_size,
        pre_selected_nct_ids=None,
        synonyms=main_conditions,
        other_conditions=other_conditions,
        vector_score_threshold=config["search"]["vector_score_threshold"],
    )

    nct_ids = [trial.get("nct_id") for trial in trials if trial.get("nct_id")]
    first_level_scores = {
        trial.get("nct_id"): score
        for trial, score in zip(trials, scores)
        if trial.get("nct_id")
    }

    write_text_file([str(nid) for nid in nct_ids], f"{output_folder}/nct_ids.txt")
    write_json_file(first_level_scores, f"{output_folder}/first_level_scores.json")
    ranked_lines = ["nct_id\tscore"] + [
        f"{nid}\t{first_level_scores[nid]:.8g}"
        for nid in nct_ids
        if nid in first_level_scores
    ]
    write_text_file(ranked_lines, f"{output_folder}/first_level_ranked.tsv")

    if explain_filter_misses:
        fb = build_eligibility_filters(
            age_for_query, sex, overall_status, None
        )
        explain_path = f"{output_folder}/first_level_filter_explain.json"
        payload = explain_first_level_filter_misses(
            es_client,
            index_name,
            fb=fb,
            retrieved_nct_ids=[str(x) for x in nct_ids if x],
            max_trials_first_level=search_size,
            sex=sex,
            overall_status=overall_status,
            pre_selected_nct_ids=None,
            patient_age_raw=age_input,
            vector_score_threshold=float(
                config.get("search", {}).get("vector_score_threshold", 0.5)
            ),
        )
        write_json_file(payload, explain_path)
        logger.info("First-level filter explain written: %s", explain_path)

    logger.info(
        f"First-level search complete: {len(nct_ids)} trial IDs saved "
        f"(scores: first_level_scores.json, first_level_ranked.tsv)."
    )
    return (
        nct_ids,
        main_conditions,
        other_conditions,
        expanded_sentences,
        first_level_scores,
    )


def _write_second_level_trial_score_artifacts(
    output_folder: str,
    second_level_mode: str,
    *,
    top_n: int,
    num_top_txt: int,
    trials_ranked_by_combined: List[Dict[str, Any]],
) -> None:
    """Persist trial-level second-stage scores (aligned with run_second_level_search)."""
    payload = {
        "second_level_search_mode": second_level_mode,
        "max_trials_second_level_cap": top_n,
        "num_lines_written_to_top_trials_txt": num_top_txt,
        "trials_ranked_by_combined": trials_ranked_by_combined,
    }
    json_path = f"{output_folder}/second_level_trial_scores.json"
    write_json_file(payload, json_path)
    header = (
        "rank_combined\tnct_id\tfirst_level_score\tsecond_level_score\t"
        "combined_score\trank_second_level\tselected_for_top_trials_txt"
    )
    lines = [header]
    for row in trials_ranked_by_combined:
        r2 = row.get("rank_second_level")
        r2s = "" if r2 is None else str(r2)
        lines.append(
            f"{row['rank_combined']}\t{row['nct_id']}\t{row['first_level_score']:.8g}\t"
            f"{row['second_level_score']:.8g}\t{row['combined_score']:.8g}\t{r2s}\t"
            f"{row['selected_for_top_trials_txt']}"
        )
    write_text_file(lines, f"{output_folder}/second_level_trial_scores.tsv")

    # Same order as top_trials.txt (first num_top_txt rows by combined rank).
    top_lines = [
        "rank_in_top_trials\tnct_id\tsecond_level_score\tfirst_level_score\t"
        "combined_score\trank_second_level\trank_combined"
    ]
    top_slice = (
        trials_ranked_by_combined[:num_top_txt] if num_top_txt > 0 else []
    )
    for i, row in enumerate(top_slice, start=1):
        r2 = row.get("rank_second_level")
        r2s = "" if r2 is None else str(r2)
        top_lines.append(
            f"{i}\t{row['nct_id']}\t{row['second_level_score']:.8g}\t"
            f"{row['first_level_score']:.8g}\t{row['combined_score']:.8g}\t{r2s}\t"
            f"{row['rank_combined']}"
        )
    write_text_file(top_lines, f"{output_folder}/top_trials_scored.tsv")


def run_second_level_search(
    output_folder: str,
    nct_ids: List[str],
    main_conditions: List[str],
    other_conditions: List[str],
    expanded_sentences: List[str],
    gemma_retriever: SecondStageRetriever,
    first_level_scores: Dict,
    config: Dict,
) -> Tuple:
    top_trials_path = f"{output_folder}/top_trials.txt"
    second_level_mode = config.get("search", {}).get(
        "second_level_search_mode", "hybrid"
    )
    if not nct_ids:
        logger.warning("No candidate trial IDs available; skipping second-level search.")
        write_text_file([], top_trials_path)
        _write_second_level_trial_score_artifacts(
            output_folder,
            second_level_mode,
            top_n=0,
            num_top_txt=0,
            trials_ranked_by_combined=[],
        )
        return [], top_trials_path

    queries = list(set(main_conditions + other_conditions + expanded_sentences))[:10]
    if not queries:
        logger.warning("No search queries available; skipping second-level search.")
        write_text_file([], top_trials_path)
        _write_second_level_trial_score_artifacts(
            output_folder,
            second_level_mode,
            top_n=0,
            num_top_txt=0,
            trials_ranked_by_combined=[],
        )
        return [], top_trials_path
    if second_level_mode == "all_rerank":
        logger.info(
            "Running second-level all-criteria reranking with %d queries ...",
            len(queries),
        )
    else:
        logger.info(
            "Running second-level %s retrieval with %d queries ...",
            second_level_mode,
            len(queries),
        )

    if queries:
        synonyms = gemma_retriever.get_synonyms(queries[0])
        queries.extend(synonyms[:3])

    top_n = min(len(nct_ids), config["search"].get("max_trials_second_level", 100))
    second_level_results = gemma_retriever.retrieve_and_rank(
        queries, nct_ids, top_n=top_n
    )

    combined_scores: Dict[str, float] = {}
    for trial in second_level_results:
        trial_id = trial["nct_id"]
        second_score = float(trial["score"])
        try:
            first_score = float(first_level_scores.get(trial_id, 0))
        except (TypeError, ValueError):
            first_score = 0.0
        combined_scores[trial_id] = first_score + second_score

    sorted_trials = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    num_top = max(1, min(len(sorted_trials) // 3, top_n))
    semi_final_trials = sorted_trials[:num_top]
    selected_ids = {trial_id for trial_id, _ in semi_final_trials}

    rank_second_level = {
        str(t["nct_id"]): idx for idx, t in enumerate(second_level_results, start=1)
    }
    second_level_only = {
        str(t["nct_id"]): float(t["score"]) for t in second_level_results
    }

    trials_ranked_by_combined: List[Dict[str, Any]] = []
    for rank_c, (trial_id, combined) in enumerate(sorted_trials, start=1):
        tid = str(trial_id)
        try:
            fs = float(first_level_scores.get(tid, 0))
        except (TypeError, ValueError):
            fs = 0.0
        ss = float(second_level_only.get(tid, 0.0))
        trials_ranked_by_combined.append(
            {
                "rank_combined": rank_c,
                "rank_second_level": rank_second_level.get(tid),
                "nct_id": tid,
                "first_level_score": fs,
                "second_level_score": ss,
                "combined_score": float(combined),
                "selected_for_top_trials_txt": tid in selected_ids,
            }
        )

    write_text_file([trial_id for trial_id, _ in semi_final_trials], top_trials_path)

    _write_second_level_trial_score_artifacts(
        output_folder,
        second_level_mode,
        top_n=top_n,
        num_top_txt=len(semi_final_trials),
        trials_ranked_by_combined=trials_ranked_by_combined,
    )

    logger.info(
        "Second-level retrieval and ranking complete. Top trials saved; "
        "scores: second_level_trial_scores.json/.tsv (all stage-2 candidates), "
        "top_trials_scored.tsv (same order as top_trials.txt)."
    )
    return semi_final_trials, top_trials_path


def run_rag_processing(
    output_folder: str,
    top_trials_file: str,
    patient_info: Dict,
    model,
    tokenizer,
    config: Dict,
):
    top_trials = read_text_file(top_trials_file)
    if not top_trials:
        logger.error("No top trials available for RAG processing.")
        return

    top_trials = top_trials[: config["rag"].get("max_trials_rag", 20)]
    patient_profile = patient_info.get("split_raw_description", [])
    if not patient_profile:
        logger.error("No patient profile available for RAG processing.")
        return

    cot_backend = config.get("cot_backend", "default")
    use_vllm = cot_backend == "vllm"

    if use_vllm:
        logger.info("Using vLLM backend for CoT reasoning")

        vllm_cfg = config.get("vllm", {})

        vllm_loader = _resolve_vllm_loader()
        vllm_engine, vllm_tokenizer, lora_request = vllm_loader(
            model_config=config.get("model", {}),
            vllm_cfg=vllm_cfg,
        )

        rag_processor = BatchTrialProcessorVLLM(
            llm=vllm_engine,  # type: ignore
            tokenizer=vllm_tokenizer,
            batch_size=vllm_cfg.get("batch_size", 16),
            use_cot=config.get("use_cot_reasoning", True),
            max_new_tokens=vllm_cfg.get("max_new_tokens", 5000),
            temperature=vllm_cfg.get("temperature", 0.0),
            top_p=vllm_cfg.get("top_p", 1.0),
            seed=vllm_cfg.get("seed", 1234),
            length_bucket=vllm_cfg.get("length_bucket", True),
            lora_request=lora_request,
        )
    else:
        logger.info("Using default (HuggingFace) backend for CoT reasoning")

        batch_size = min(config["rag"]["batch_size"] * 2, 8)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        rag_processor = BatchTrialProcessor(
            model,
            tokenizer,
            device=config["global"]["device"],
            batch_size=batch_size,
        )

    rag_processor.process_trials(
        nct_ids=top_trials,
        json_folder=config["paths"]["trials_json_folder"],
        output_folder=output_folder,
        patient_profile=patient_profile,
    )
    write_json_file({"status": "done"}, f"{output_folder}/rag_output.json")
    logger.info("RAG-based trial matching complete.")


# ── CLI argument handling ───────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TrialMatchAI — match patient phenopackets to clinical trials.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Run with defaults from config.json
  python -m Matcher.main

  # Specify patient data folder and output directory
  python -m Matcher.main --patients-dir ../patients/ --output-dir ../results/

  # Use a different base model with 4-bit quantization
  python -m Matcher.main --base-model microsoft/phi-4 --load-in-4bit

  # Run with vLLM backend and tensor parallelism
  python -m Matcher.main --cot-backend vllm --tensor-parallel-size 2

  # CPU-only run (no GPU)
  python -m Matcher.main --device cpu

  # Override Elasticsearch connection
  python -m Matcher.main --es-host https://es-server:9200 --es-password secret

  # Override Elasticsearch index names (trial + eligibility criteria)
  python -m Matcher.main --es-index-trials my_trials --es-index-trials-eligibility my_eic

  # Custom config file
  python -m Matcher.main --config my_config.json
""",
    )

    # ── general ──
    general = parser.add_argument_group("general")
    general.add_argument(
        "--config",
        default="Matcher/config/config.json",
        help="Path to the JSON configuration file (default: Matcher/config/config.json)",
    )
    general.add_argument(
        "--device",
        default=None,
        help="Compute device: GPU index (0, 1, …) or 'cpu' (default: from config)",
    )

    # ── patient data ──
    patient = parser.add_argument_group("patient data")
    patient.add_argument(
        "--patients-dir",
        default=None,
        help="Directory containing patient phenopacket JSON files (default: from config)",
    )
    patient.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write pipeline results (default: from config)",
    )
    patient.add_argument(
        "--trials-json-folder",
        default=None,
        help="Folder with processed trial JSONs for RAG (default: from config)",
    )

    # ── model ──
    model_grp = parser.add_argument_group("model")
    model_grp.add_argument(
        "--base-model",
        default=None,
        help="HuggingFace model name or local path for the reasoning LLM "
        "(default: from config, e.g. microsoft/phi-4)",
    )
    model_grp.add_argument(
        "--cot-adapter-path",
        default=None,
        help="Path to the CoT LoRA adapter (default: from config)",
    )
    model_grp.add_argument(
        "--reranker-model-path",
        default=None,
        help="HuggingFace model name for the reranker (default: from config)",
    )
    model_grp.add_argument(
        "--reranker-adapter-path",
        default=None,
        help="Path to the reranker LoRA adapter (default: from config)",
    )

    # ── quantization ──
    quant = parser.add_argument_group("quantization")
    quant.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=None,
        help="Enable 4-bit quantization via bitsandbytes",
    )
    quant.add_argument(
        "--no-4bit",
        action="store_true",
        default=False,
        help="Disable 4-bit quantization (full precision or fp16)",
    )
    quant.add_argument(
        "--quant-type",
        choices=["nf4", "fp4"],
        default=None,
        help="4-bit quantization type (default: nf4)",
    )
    quant.add_argument(
        "--double-quant",
        action="store_true",
        default=None,
        help="Enable double quantization for 4-bit (saves memory)",
    )

    # ── embedder ──
    embed = parser.add_argument_group("embedder")
    embed.add_argument(
        "--embedder-model",
        default=None,
        help="Sentence embedding model (default: BAAI/bge-m3)",
    )
    embed.add_argument(
        "--embedder-pooling",
        choices=["mean", "cls"],
        default=None,
        help="Embedding pooling strategy (default: mean)",
    )
    embed.add_argument(
        "--embedder-max-length",
        type=int,
        default=None,
        help="Max token length for embedding (default: 512)",
    )
    embed.add_argument(
        "--embedder-fp16",
        action="store_true",
        default=None,
        help="Use FP16 for the embedder model",
    )

    # ── CoT / RAG backend ──
    cot = parser.add_argument_group("CoT / RAG backend")
    cot.add_argument(
        "--cot-backend",
        choices=["default", "vllm"],
        default=None,
        help="Backend for chain-of-thought reasoning: 'default' (HuggingFace) or 'vllm' "
        "(default: from config)",
    )
    cot.add_argument(
        "--use-cot-reasoning",
        action="store_true",
        default=None,
        help="Enable chain-of-thought reasoning (default: True)",
    )
    cot.add_argument(
        "--no-cot-reasoning",
        action="store_true",
        default=False,
        help="Disable chain-of-thought reasoning",
    )
    cot.add_argument(
        "--rag-batch-size",
        type=int,
        default=None,
        help="Batch size for RAG processing (default: from config)",
    )
    cot.add_argument(
        "--max-trials-rag",
        type=int,
        default=None,
        help="Max trials sent through RAG (default: 20)",
    )

    # ── vLLM ──
    vllm_grp = parser.add_argument_group("vLLM options (only when --cot-backend vllm)")
    vllm_grp.add_argument(
        "--vllm-batch-size",
        type=int,
        default=None,
        help="vLLM batch size (default: 100)",
    )
    vllm_grp.add_argument(
        "--vllm-max-new-tokens",
        type=int,
        default=None,
        help="Max new tokens for vLLM generation (default: 5000)",
    )
    vllm_grp.add_argument(
        "--vllm-temperature",
        type=float,
        default=None,
        help="Sampling temperature for vLLM (default: 0.0 = greedy)",
    )
    vllm_grp.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=None,
        help="Fraction of GPU memory for vLLM (default: 0.5)",
    )
    vllm_grp.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Tensor parallelism degree for vLLM (default: 1)",
    )
    vllm_grp.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help="Max context length for vLLM engine (default: 8192)",
    )
    vllm_grp.add_argument(
        "--vllm-max-num-seqs",
        type=int,
        default=None,
        help="Max concurrent sequences for vLLM scheduler/warmup (default: 128)",
    )
    vllm_grp.add_argument(
        "--vllm-enforce-eager",
        action="store_true",
        help="Disable CUDA graph capture for vLLM to reduce startup memory pressure",
    )

    # ── search ──
    search = parser.add_argument_group("search")
    search.add_argument(
        "--vector-score-threshold",
        type=float,
        default=None,
        help=(
            "First-level (trial index): minimum combined vector score in hybrid/vector "
            "script_score (default: 0.5)"
        ),
    )
    search.add_argument(
        "--second-level-vector-score-threshold",
        type=float,
        default=None,
        help=(
            "Second-level (criteria index): minimum normalized cosine score for vector/hybrid "
            "script_score (default: 0.5)"
        ),
    )
    search.add_argument(
        "--second-level-aggregate-score-threshold",
        type=float,
        default=None,
        help=(
            "Second-level: minimum per-criterion llm_score (after inclusion/exclusion weights) "
            "to count toward aggregate_to_trials per trial (default: 0.5). Lower values keep "
            "more trials after rerank."
        ),
    )
    search.add_argument(
        "--max-trials-first-level",
        type=int,
        default=None,
        help="Max trials returned by first-level search (default: 300)",
    )
    search.add_argument(
        "--max-trials-second-level",
        type=int,
        default=None,
        help="Max trials returned by second-level search (default: 100)",
    )
    search.add_argument(
        "--skip-first-level",
        action="store_true",
        default=None,
        help="Skip first-level retrieval and use all trial IDs from --trials-json-folder for second-level reranking.",
    )
    search.add_argument(
        "--resume-from-second-level",
        action="store_true",
        default=None,
        help=(
            "Skip first/second-level search and start from existing "
            "<output_dir>/<patient_id>/top_trials.txt for CoT/RAG."
        ),
    )
    search.add_argument(
        "--second-level-search-mode",
        choices=["hybrid", "bm25", "vector", "all_rerank"],
        default=None,
        help=(
            "Second-level criteria selection mode: hybrid/bm25/vector retrieval, "
            "or all_rerank to skip retrieval and rerank all criteria from the "
            "candidate trials."
        ),
    )
    search.add_argument(
        "--explain-first-level-filter-misses",
        action="store_true",
        default=None,
        help=(
            "After first-level search, write first_level_filter_explain.json: "
            "per-trial reasons for failing age/gender/status filters vs "
            "passed-filters-but-not-in-top-K (ranking cutoff)."
        ),
    )

    # ── Elasticsearch ──
    es = parser.add_argument_group("Elasticsearch")
    es.add_argument("--es-host", default=None, help="Elasticsearch host URL")
    es.add_argument("--es-username", default=None, help="Elasticsearch username")
    es.add_argument("--es-password", default=None, help="Elasticsearch password")
    es.add_argument(
        "--es-ca-certs", default=None, help="Path to Elasticsearch CA cert"
    )
    es.add_argument(
        "--es-auto-start",
        action="store_true",
        default=None,
        help="Auto-start Elasticsearch if not running",
    )
    es.add_argument(
        "--es-index-trials",
        default=None,
        help="Elasticsearch index for trial documents (default: from config)",
    )
    es.add_argument(
        "--es-index-trials-eligibility",
        default=None,
        help="Elasticsearch index for eligibility criteria (default: from config)",
    )

    return parser


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Merge CLI arguments into the loaded config. CLI wins over config.json."""

    # ── device ──
    if args.device is not None:
        val = args.device
        try:
            val = int(val)
        except ValueError:
            pass
        config["global"]["device"] = val

    # ── paths ──
    if args.patients_dir is not None:
        config["paths"]["patients_dir"] = args.patients_dir
    if args.output_dir is not None:
        config["paths"]["output_dir"] = args.output_dir
    if args.trials_json_folder is not None:
        config["paths"]["trials_json_folder"] = args.trials_json_folder

    # ── model ──
    if args.base_model is not None:
        config["model"]["base_model"] = args.base_model
    if args.cot_adapter_path is not None:
        config["model"]["cot_adapter_path"] = args.cot_adapter_path
    if args.reranker_model_path is not None:
        config["model"]["reranker_model_path"] = args.reranker_model_path
    if args.reranker_adapter_path is not None:
        config["model"]["reranker_adapter_path"] = args.reranker_adapter_path

    # ── quantization ──
    if args.no_4bit:
        config["model"]["quantization"]["load_in_4bit"] = False
    elif args.load_in_4bit is True:
        config["model"]["quantization"]["load_in_4bit"] = True
    if args.quant_type is not None:
        config["model"]["quantization"]["bnb_4bit_quant_type"] = args.quant_type
    if args.double_quant is True:
        config["model"]["quantization"]["bnb_4bit_use_double_quant"] = True

    # ── embedder ──
    if args.embedder_model is not None:
        config["embedder"]["model_name"] = args.embedder_model
    if args.embedder_pooling is not None:
        config["embedder"]["pooling"] = args.embedder_pooling
    if args.embedder_max_length is not None:
        config["embedder"]["max_length"] = args.embedder_max_length
    if args.embedder_fp16 is True:
        config["embedder"]["use_fp16"] = True

    # ── CoT / RAG ──
    if args.cot_backend is not None:
        config["cot_backend"] = args.cot_backend
    if args.no_cot_reasoning:
        config["use_cot_reasoning"] = False
    elif args.use_cot_reasoning is True:
        config["use_cot_reasoning"] = True
    if args.rag_batch_size is not None:
        config["rag"]["batch_size"] = args.rag_batch_size
    if args.max_trials_rag is not None:
        config["rag"]["max_trials_rag"] = args.max_trials_rag

    # ── vLLM ──
    if args.vllm_batch_size is not None:
        config["vllm"]["batch_size"] = args.vllm_batch_size
    if args.vllm_max_new_tokens is not None:
        config["vllm"]["max_new_tokens"] = args.vllm_max_new_tokens
    if args.vllm_temperature is not None:
        config["vllm"]["temperature"] = args.vllm_temperature
    if args.vllm_gpu_memory_utilization is not None:
        config["vllm"]["gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
    if args.tensor_parallel_size is not None:
        config["vllm"]["tensor_parallel_size"] = args.tensor_parallel_size
    if args.vllm_max_model_len is not None:
        config["vllm"]["max_model_len"] = args.vllm_max_model_len
    if args.vllm_max_num_seqs is not None:
        config["vllm"]["max_num_seqs"] = args.vllm_max_num_seqs
    if args.vllm_enforce_eager:
        config["vllm"]["enforce_eager"] = True

    # ── search ──
    if args.vector_score_threshold is not None:
        config["search"]["vector_score_threshold"] = args.vector_score_threshold
    if args.second_level_vector_score_threshold is not None:
        config["search"]["second_level_vector_score_threshold"] = (
            args.second_level_vector_score_threshold
        )
    if args.second_level_aggregate_score_threshold is not None:
        config["search"]["second_level_aggregate_score_threshold"] = (
            args.second_level_aggregate_score_threshold
        )
    if args.max_trials_first_level is not None:
        config["search"]["max_trials_first_level"] = args.max_trials_first_level
    if args.max_trials_second_level is not None:
        config["search"]["max_trials_second_level"] = args.max_trials_second_level
    if args.skip_first_level is True:
        config["search"]["skip_first_level"] = True
    if args.resume_from_second_level is True:
        config["search"]["resume_from_second_level"] = True
    if args.second_level_search_mode is not None:
        config["search"]["second_level_search_mode"] = args.second_level_search_mode
    if args.explain_first_level_filter_misses is True:
        config["search"]["explain_first_level_filter_misses"] = True

    # ── Elasticsearch ──
    if args.es_host is not None:
        config["elasticsearch"]["host"] = args.es_host
    if args.es_username is not None:
        config["elasticsearch"]["username"] = args.es_username
    if args.es_password is not None:
        config["elasticsearch"]["password"] = args.es_password
    if args.es_ca_certs is not None:
        config["paths"]["docker_certs"] = args.es_ca_certs
    if args.es_auto_start is True:
        config["elasticsearch"]["auto_start"] = True
    if args.es_index_trials is not None:
        config["elasticsearch"]["index_trials"] = args.es_index_trials
    if args.es_index_trials_eligibility is not None:
        config["elasticsearch"]["index_trials_eligibility"] = (
            args.es_index_trials_eligibility
        )

    return config


# ── Main pipeline ───────────────────────────────────────────────────────


def main_pipeline(config: Dict[str, Any]):
    logger.info("Starting TrialMatchAI pipeline...")
    paths = config["paths"]
    create_directory(paths["output_dir"])

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)

    initialize_biomedner_services(config)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*quantization_config.*", category=UserWarning
        )
        model, tokenizer = load_model_and_tokenizer(
            config["model"], config["global"]["device"]
        )

    if tokenizer.pad_token is None:  # type: ignore
        tokenizer.pad_token = tokenizer.eos_token  # type: ignore
        if hasattr(model.config, "pad_token_id") and model.config.pad_token_id is None:  # type: ignore
            model.config.pad_token_id = tokenizer.pad_token_id  # type: ignore

    if config["global"]["device"] != "cpu" and torch.cuda.is_available():
        q = config.get("model", {}).get("quantization", {})
        quantized = bool(q.get("load_in_4bit", False)) or getattr(
            model, "is_loaded_in_4bit", False
        ) or getattr(model, "is_loaded_in_8bit", False)
        if not quantized:
            model = model.half()  # type: ignore

    embedder_cfg = config.get("embedder", {})
    embedder = TextEmbedder(
        TextEmbedderConfig(
            model_name=embedder_cfg.get("model_name", "BAAI/bge-m3"),
            pooling=embedder_cfg.get("pooling", "mean"),
            max_length=embedder_cfg.get("max_length", 512),
            batch_size=embedder_cfg.get("batch_size", 32),
            use_gpu=embedder_cfg.get("use_gpu", True),
            use_fp16=embedder_cfg.get("use_fp16", False),
            normalize=embedder_cfg.get("normalize", True),
        )
    )
    bio_med_ner = BioMedNER(**config["bio_med_ner"])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*quantization_config.*", category=UserWarning
        )
        llm_reranker = LLMReranker(
            model_path=config["model"]["reranker_model_path"],
            adapter_path=config["model"]["reranker_adapter_path"],
            device=config["global"]["device"],
            batch_size=config["rag"]["batch_size"] * 2,
        )

    es_client = Elasticsearch(
        hosts=[config["elasticsearch"]["host"]],
        ca_certs=paths["docker_certs"],
        basic_auth=(
            config["elasticsearch"]["username"],
            config["elasticsearch"]["password"],
        ),
        request_timeout=config["elasticsearch"]["request_timeout"],
        retry_on_timeout=config["elasticsearch"]["retry_on_timeout"],
    )
    if not ensure_elasticsearch(es_client, config):
        return

    gemma_retriever = SecondStageRetriever(
        es_client=es_client,
        llm_reranker=llm_reranker,
        embedder=embedder,
        index_name=config["elasticsearch"]["index_trials_eligibility"],
        bio_med_ner=bio_med_ner,
        search_mode=config.get("search", {}).get(
            "second_level_search_mode", "hybrid"
        ),
        second_level_vector_score_threshold=config.get("search", {}).get(
            "second_level_vector_score_threshold", 0.5
        ),
        second_level_aggregate_score_threshold=config.get("search", {}).get(
            "second_level_aggregate_score_threshold", 0.5
        ),
    )

    patient_folder = Path(paths["patients_dir"])
    if not patient_folder.exists():
        logger.error("Patients folder not found: %s", patient_folder)
        return
    phenopacket_files = sorted(
        [p for p in patient_folder.iterdir() if p.suffix == ".json"]
    )
    if not phenopacket_files:
        logger.warning("No patient files found in %s", patient_folder)
        return

    logger.info("Found %d patient phenopacket(s) in %s", len(phenopacket_files), patient_folder)
    skip_first_level = bool(config.get("search", {}).get("skip_first_level", False))
    resume_from_second_level = bool(
        config.get("search", {}).get("resume_from_second_level", False)
    )
    all_trial_ids: List[str] = []
    if skip_first_level and not resume_from_second_level:
        all_trial_ids = load_all_trial_ids_from_folder(paths["trials_json_folder"])
        if not all_trial_ids:
            logger.error(
                "skip_first_level is enabled but no candidate trials were loaded."
            )
            return

    for phenopacket_path in phenopacket_files:
        patient_id = phenopacket_path.stem
        token = set_request_id(patient_id)
        output_folder = Path(paths["output_dir"]) / patient_id
        create_directory(str(output_folder))

        input_file = str(phenopacket_path)
        output_file = str(output_folder / "keywords.json")

        try:
            keywords: Optional[Dict[str, Any]] = None
            if Path(output_file).exists():
                existing_keywords = Keywords.model_validate(
                    read_json_file(output_file)
                ).model_dump()
                has_query_terms = any(
                    existing_keywords.get(field)
                    for field in (
                        "main_conditions",
                        "other_conditions",
                        "expanded_sentences",
                    )
                )
                if has_query_terms or resume_from_second_level:
                    keywords = existing_keywords
                    logger.info(
                        "Loaded existing keywords from %s; skipping phenopacket processing.",
                        output_file,
                    )
                else:
                    logger.warning(
                        "Existing keywords at %s are empty; regenerating keywords.",
                        output_file,
                    )

            if keywords is None:
                if resume_from_second_level:
                    logger.info(
                        "Resume mode: keywords not found; running phenopacket processing."
                    )
                with log_timing(logger, "Phenopacket processing"):
                    with torch.no_grad():
                        process_phenopacket(
                            input_file, output_file, model=model, tokenizer=tokenizer
                        )
                keywords = Keywords.model_validate(
                    read_json_file(output_file)
                ).model_dump()

            patient_info = Phenopacket.model_validate(
                read_json_file(input_file)
            ).model_dump()
            patient_info["split_raw_description"] = keywords.get(
                "expanded_sentences", []
            )
            _raw_age = patient_info.get("age")
            if _raw_age in (None, "", "all", "ALL", "All"):
                _inferred = infer_patient_age_years_from_phenopacket(patient_info)
                if _inferred is not None:
                    patient_info["age"] = _inferred

            if resume_from_second_level:
                top_trials_path = str(output_folder / "top_trials.txt")
                top_trials = read_text_file(top_trials_path)
                if not top_trials:
                    logger.error(
                        "Resume mode enabled but top trials not found or empty: %s",
                        top_trials_path,
                    )
                    continue
                logger.info(
                    "Resuming from second-level outputs; skipping first/second-level search."
                )
            elif skip_first_level:
                nct_ids = all_trial_ids
                main_conditions = keywords.get("main_conditions", [])
                other_conditions = keywords.get("other_conditions", [])
                expanded_sentences = keywords.get("expanded_sentences", [])
                first_level_scores = {}
                write_text_file(
                    [str(nid) for nid in nct_ids], f"{output_folder}/nct_ids.txt"
                )
                write_json_file({}, f"{output_folder}/first_level_scores.json")
                logger.info(
                    "Skipping first-level search; using %d trial IDs from trials_json_folder.",
                    len(nct_ids),
                )
            else:
                with log_timing(logger, "First-level search"):
                    with torch.no_grad():
                        result = run_first_level_search(
                            keywords,
                            str(output_folder),
                            patient_info,
                            bio_med_ner,
                            embedder,
                            config,
                            es_client,
                            explain_filter_misses=config["search"].get(
                                "explain_first_level_filter_misses", False
                            ),
                        )
                if not result:
                    logger.error("First-level search failed for %s", patient_id)
                    continue

                (
                    nct_ids,
                    main_conditions,
                    other_conditions,
                    expanded_sentences,
                    first_level_scores,
                ) = result

            if not resume_from_second_level:
                with log_timing(logger, "Second-level search"):
                    with torch.no_grad():
                        _, top_trials_path = run_second_level_search(
                            str(output_folder),
                            nct_ids,
                            main_conditions,
                            other_conditions,
                            expanded_sentences,
                            gemma_retriever,
                            first_level_scores,
                            config,
                        )

            with log_timing(logger, "RAG processing"):
                with torch.no_grad():
                    run_rag_processing(
                        str(output_folder),
                        top_trials_path,
                        patient_info,
                        model,
                        tokenizer,
                        config,
                    )

            with log_timing(logger, "Final ranking"):
                trial_data = load_trial_data(str(output_folder))
                ranked_trials = rank_trials(trial_data)
                save_ranked_trials(
                    ranked_trials, str(output_folder / "ranked_trials.json")
                )

            logger.info("Pipeline completed for patient %s", patient_id)
        except Exception:
            logger.exception("Pipeline failed for patient %s", patient_id)
            continue
        finally:
            reset_request_id(token)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    main_pipeline(config)

