#!/usr/bin/env python3
"""Evaluate clinical-trial predictions against patient-level ground truth.

Ground truth format (single object, list of objects, or directory of JSON files):
{
  "patient_id": "P001",
  "patient_file": "../example/processed_patients/P001_xxx.json",
  "matched_trials": [
    {"trial_id": "CTR20241474", "matched": true},
    ...
  ]
}

Prediction format supports either:

1) Ranked format (found recursively under --predictions-dir):
results/<patient_name>/ranked_trials.json
{
  "RankedTrials": [{"TrialID": "NCT...", "Score": 1.0}, ...]
}

2) Patient record format (same style as ground truth):
{
  "patient_id": "P001",
  "patient_file": ".../P001_xxx.json",
  "matched_trials": [{"trial_id": "...", "matched": true}, ...]
}
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass
class PatientGroundTruth:
    patient_id: str
    aliases: Set[str]
    relevant_trial_ids: Set[str]


_TRIAL_ID_RE = re.compile(r"^(NCT\d+|CTR\d+)$", re.IGNORECASE)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _norm_id(value: Any) -> str:
    return _normalize_text(value).upper()


def _extract_trial_id(item: Any) -> Optional[str]:
    if isinstance(item, str):
        nid = _norm_id(item)
        if nid and _TRIAL_ID_RE.match(nid):
            return nid
        return None

    if isinstance(item, dict):
        for key in ("TrialID", "trial_id", "trialId", "nct_id", "id"):
            if key in item:
                nid = _norm_id(item.get(key))
                if nid and _TRIAL_ID_RE.match(nid):
                    return nid
    return None


def _extract_ranked_list(raw: Any) -> List[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("RankedTrials", "ranked_trials", "ranked", "trials"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
    return []


def _extract_ranked_ids_above_score(
    raw_prediction: Any, min_pred_score: float
) -> List[str]:
    ranked = _extract_ranked_list(raw_prediction)
    ids: List[str] = []
    seen: Set[str] = set()
    for item in ranked:
        if not isinstance(item, dict):
            continue
        tid = _extract_trial_id(item)
        if not tid or tid in seen:
            continue
        try:
            score = float(item.get("Score", item.get("score", 0.0)))
        except Exception:
            score = 0.0
        if score < min_pred_score:
            continue
        seen.add(tid)
        ids.append(tid)
    return ids


def _extract_prediction_ids_above_score(
    raw_prediction: Any, min_pred_score: float
) -> List[str]:
    if isinstance(raw_prediction, dict):
        matched_trials = raw_prediction.get("matched_trials")
        if isinstance(matched_trials, list):
            ids: List[str] = []
            seen: Set[str] = set()
            for item in matched_trials:
                if not isinstance(item, dict):
                    continue
                if item.get("matched") is False:
                    continue
                tid = _extract_trial_id(item)
                if not tid or tid in seen:
                    continue
                try:
                    score = float(item.get("match_score", item.get("Score", 0.0)))
                except Exception:
                    score = 0.0
                if score < min_pred_score:
                    continue
                seen.add(tid)
                ids.append(tid)
            return ids

        all_assessments = raw_prediction.get("all_assessments")
        if isinstance(all_assessments, list):
            ids: List[str] = []
            seen: Set[str] = set()
            for item in all_assessments:
                if not isinstance(item, dict):
                    continue
                if item.get("matched") is False:
                    continue
                tid = _extract_trial_id(item)
                if not tid or tid in seen:
                    continue
                try:
                    score = float(item.get("match_score", item.get("Score", 0.0)))
                except Exception:
                    score = 0.0
                if score < min_pred_score:
                    continue
                seen.add(tid)
                ids.append(tid)
            return ids

    return _extract_ranked_ids_above_score(raw_prediction, min_pred_score)


def _extract_predicted_ids(raw_prediction: Any) -> List[str]:
    if isinstance(raw_prediction, dict):
        # Support prediction JSONs in the same schema as ground truth output.
        matched_trials = raw_prediction.get("matched_trials")
        if isinstance(matched_trials, list):
            ids: List[str] = []
            seen: Set[str] = set()
            for item in matched_trials:
                if not isinstance(item, dict):
                    continue
                if item.get("matched") is False:
                    continue
                tid = _extract_trial_id(item)
                if not tid or tid in seen:
                    continue
                seen.add(tid)
                ids.append(tid)
            return ids

        all_assessments = raw_prediction.get("all_assessments")
        if isinstance(all_assessments, list):
            ids = []
            seen: Set[str] = set()
            for item in all_assessments:
                if not isinstance(item, dict):
                    continue
                if item.get("matched") is False:
                    continue
                tid = _extract_trial_id(item)
                if not tid or tid in seen:
                    continue
                seen.add(tid)
                ids.append(tid)
            return ids

    ranked = _extract_ranked_list(raw_prediction)
    ids: List[str] = []
    seen: Set[str] = set()
    for item in ranked:
        tid = _extract_trial_id(item)
        if not tid or tid in seen:
            continue
        seen.add(tid)
        ids.append(tid)
    return ids


def _looks_like_patient_prediction_record(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    has_patient = bool(obj.get("patient_id") or obj.get("id") or obj.get("patient_file"))
    has_trials = isinstance(obj.get("matched_trials"), list) or isinstance(
        obj.get("all_assessments"), list
    )
    return has_patient and has_trials


def _extract_gt_relevant_ids(gt_obj: Dict[str, Any]) -> Set[str]:
    matched_trials = gt_obj.get("matched_trials")
    if not isinstance(matched_trials, list):
        return set()

    rel: Set[str] = set()
    for item in matched_trials:
        if not isinstance(item, dict):
            continue
        matched = item.get("matched")
        # If 'matched' is absent, treat matched_trials entries as relevant by default.
        if matched is False:
            continue
        tid = _extract_trial_id(item)
        if tid:
            rel.add(tid)
    return rel


def _extract_patient_aliases(gt_obj: Dict[str, Any]) -> Set[str]:
    aliases: Set[str] = set()

    patient_id = _normalize_text(gt_obj.get("patient_id") or gt_obj.get("id"))
    if patient_id:
        aliases.add(patient_id)
        aliases.add(patient_id.lower())

    patient_file = _normalize_text(gt_obj.get("patient_file"))
    if patient_file:
        pf = Path(patient_file)
        aliases.add(pf.name)
        aliases.add(pf.name.lower())
        aliases.add(pf.stem)
        aliases.add(pf.stem.lower())
        # Common naming pattern: P001_xxx_符合.json -> add P001 as alias.
        stem = pf.stem
        if "_" in stem:
            first = stem.split("_", 1)[0].strip()
            if first:
                aliases.add(first)
                aliases.add(first.lower())

    return aliases


def _iter_gt_records(source: Path) -> Iterable[Dict[str, Any]]:
    if source.is_file():
        data = _load_json(source)
        if isinstance(data, dict):
            yield data
            return
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
            return
        raise ValueError(f"Unsupported ground-truth JSON shape in: {source}")

    if source.is_dir():
        for path in sorted(source.rglob("*.json")):
            data = _load_json(path)
            if isinstance(data, dict):
                yield data
            elif isinstance(data, list):
                for obj in data:
                    if isinstance(obj, dict):
                        yield obj
        return

    raise FileNotFoundError(f"Ground truth path not found: {source}")


def load_ground_truth(source: Path) -> List[PatientGroundTruth]:
    patients: List[PatientGroundTruth] = []
    for rec in _iter_gt_records(source):
        rel = _extract_gt_relevant_ids(rec)
        aliases = _extract_patient_aliases(rec)
        patient_id = _normalize_text(rec.get("patient_id") or rec.get("id"))
        if not patient_id and aliases:
            # Prefer a stable alias if patient_id is missing.
            patient_id = sorted(aliases)[0]
        if not patient_id:
            continue
        patients.append(
            PatientGroundTruth(
                patient_id=patient_id,
                aliases=aliases | {patient_id, patient_id.lower()},
                relevant_trial_ids=rel,
            )
        )
    return patients


def load_predictions(predictions_path: Path, filename: str) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}

    paths: List[Path] = []
    if predictions_path.is_file():
        paths = [predictions_path]
    elif predictions_path.is_dir():
        paths = sorted(predictions_path.rglob(filename))
    else:
        raise FileNotFoundError(f"Predictions path not found: {predictions_path}")

    for path in paths:
        raw = _load_json(path)

        # Case A: predictions are patient records (same schema family as ground truth).
        if _looks_like_patient_prediction_record(raw):
            predicted_ids = _extract_predicted_ids(raw)
            aliases = _extract_patient_aliases(raw)
            if aliases:
                for alias in aliases:
                    mapping[alias] = predicted_ids
            continue
        if isinstance(raw, list):
            records = [obj for obj in raw if _looks_like_patient_prediction_record(obj)]
            if records:
                for rec in records:
                    predicted_ids = _extract_predicted_ids(rec)
                    aliases = _extract_patient_aliases(rec)
                    if aliases:
                        for alias in aliases:
                            mapping[alias] = predicted_ids
                continue

        # Case B: ranked_trials format keyed by folder name (legacy behavior).
        ranked_ids = _extract_predicted_ids(raw)
        patient_key = path.parent.name
        mapping[patient_key] = ranked_ids
        mapping[patient_key.lower()] = ranked_ids
        mapping[path.parent.name + ".json"] = ranked_ids
        mapping[(path.parent.name + ".json").lower()] = ranked_ids
    return mapping


def load_predictions_above_score(
    predictions_path: Path, filename: str, min_pred_score: float
) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}

    paths: List[Path] = []
    if predictions_path.is_file():
        paths = [predictions_path]
    elif predictions_path.is_dir():
        paths = sorted(predictions_path.rglob(filename))
    else:
        raise FileNotFoundError(f"Predictions path not found: {predictions_path}")

    for path in paths:
        raw = _load_json(path)

        if _looks_like_patient_prediction_record(raw):
            predicted_ids = _extract_prediction_ids_above_score(raw, min_pred_score)
            aliases = _extract_patient_aliases(raw)
            if aliases:
                for alias in aliases:
                    mapping[alias] = predicted_ids
            continue
        if isinstance(raw, list):
            records = [obj for obj in raw if _looks_like_patient_prediction_record(obj)]
            if records:
                for rec in records:
                    predicted_ids = _extract_prediction_ids_above_score(
                        rec, min_pred_score
                    )
                    aliases = _extract_patient_aliases(rec)
                    if aliases:
                        for alias in aliases:
                            mapping[alias] = predicted_ids
                continue

        ranked_ids = _extract_prediction_ids_above_score(raw, min_pred_score)
        patient_key = path.parent.name
        mapping[patient_key] = ranked_ids
        mapping[patient_key.lower()] = ranked_ids
        mapping[path.parent.name + ".json"] = ranked_ids
        mapping[(path.parent.name + ".json").lower()] = ranked_ids
    return mapping


def _relevant_ranks_within_k(
    predicted_ids: Sequence[str], relevant_ids: Set[str], k: int
) -> List[int]:
    if k <= 0:
        return []
    ranks: List[int] = []
    for i, tid in enumerate(predicted_ids[:k], start=1):
        if tid in relevant_ids:
            ranks.append(i)
    return ranks


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _unique_prediction_lists(pred_map: Dict[str, List[str]]) -> List[List[str]]:
    unique: List[List[str]] = []
    seen: Set[Tuple[str, ...]] = set()
    for ids in pred_map.values():
        key = tuple(ids)
        if key in seen:
            continue
        seen.add(key)
        unique.append(ids)
    return unique


def evaluate(
    gt_patients: Sequence[PatientGroundTruth],
    pred_map: Dict[str, List[str]],
) -> Tuple[Dict[str, float], List[Dict[str, Any]], List[str]]:
    ks = (3, 5)
    overall_recall_vals: Dict[int, List[float]] = {k: [] for k in ks}
    mrr_vals: Dict[int, List[float]] = {k: [] for k in ks}
    retrieved_ranks: Dict[int, List[int]] = {k: [] for k in ks}

    per_patient: List[Dict[str, Any]] = []
    missing_predictions: List[str] = []
    ground_truth_relevant_total = 0
    predicted_ranked_total = 0
    matched_predicted_total = 0
    unique_ground_truth_trials: Set[str] = set()
    unique_predicted_trials: Set[str] = set()
    unique_pred_lists = _unique_prediction_lists(pred_map)
    single_prediction: Optional[List[str]] = None
    if len(gt_patients) == 1 and len(unique_pred_lists) == 1:
        single_prediction = unique_pred_lists[0]

    for gt in gt_patients:
        predicted_ids: Optional[List[str]] = None
        for alias in gt.aliases:
            if alias in pred_map:
                predicted_ids = pred_map[alias]
                break
        if predicted_ids is None and single_prediction is not None:
            # Single-gt + single-prediction fallback: auto-pair them.
            predicted_ids = single_prediction

        if predicted_ids is None:
            missing_predictions.append(gt.patient_id)
            continue

        rel = gt.relevant_trial_ids
        matched_predicted_trials = [tid for tid in predicted_ids if tid in rel]
        ground_truth_relevant_total += len(rel)
        predicted_ranked_total += len(predicted_ids)
        matched_predicted_total += len(matched_predicted_trials)
        unique_ground_truth_trials.update(rel)
        unique_predicted_trials.update(predicted_ids)

        patient_metrics: Dict[str, Any] = {}
        for k in ks:
            ranks = _relevant_ranks_within_k(predicted_ids, rel, k)
            # Table-style Overall Recall@k: patient hit rate within top-k.
            overall_recall_k = 1.0 if ranks else 0.0
            # Table-style MRR@k: reciprocal rank of first GT in top-k, else 0.
            mrr_k = 0.0 if not ranks else 1.0 / min(ranks)
            overall_recall_vals[k].append(overall_recall_k)
            mrr_vals[k].append(mrr_k)
            retrieved_ranks[k].extend(ranks)
            patient_metrics[f"overall_recall@{k}"] = overall_recall_k
            patient_metrics[f"mrr@{k}"] = mrr_k
            patient_metrics[f"retrieved_gt_ranks@{k}"] = ranks

        # Backward-compatible aliases.
        patient_metrics["top3"] = patient_metrics["overall_recall@3"]
        patient_metrics["top5"] = patient_metrics["overall_recall@5"]
        patient_metrics["reciprocal_rank"] = patient_metrics["mrr@5"]
        patient_metrics["recall"] = patient_metrics["overall_recall@5"]

        per_patient.append(
            {
                "patient_id": gt.patient_id,
                "ground_truth_relevant_count": len(rel),
                "predicted_count": len(predicted_ids),
                "matched_predicted_trials": matched_predicted_trials,
                "matched_predicted_count": len(matched_predicted_trials),
                **patient_metrics,
            }
        )

    summary = {
        "evaluated_patients": float(len(per_patient)),
        "ground_truth_patients": float(len(gt_patients)),
        # Per-patient sums (same trial in two patients counts twice).
        "ground_truth_relevant_total": float(ground_truth_relevant_total),
        "predicted_ranked_total": float(predicted_ranked_total),
        "matched_predicted_total": float(matched_predicted_total),
        # Distinct trial IDs across evaluated patients only.
        "unique_ground_truth_trials": float(len(unique_ground_truth_trials)),
        "unique_predicted_trials": float(len(unique_predicted_trials)),
    }
    for k in ks:
        summary[f"overall_recall@{k}"] = mean(overall_recall_vals[k])
        summary[f"mrr@{k}"] = mean(mrr_vals[k])
        summary[f"mean_average_rank@{k}"] = mean(
            [float(v) for v in retrieved_ranks[k]]
        )
        summary[f"retrieved_gt_count@{k}"] = float(len(retrieved_ranks[k]))

    # Backward-compatible aliases.
    summary["top3"] = summary["overall_recall@3"]
    summary["top5"] = summary["overall_recall@5"]
    summary["mrr"] = summary["mrr@5"]
    summary["recall"] = summary["overall_recall@5"]
    return summary, per_patient, missing_predictions


def compute_threshold_match_stats(
    gt_patients: Sequence[PatientGroundTruth],
    pred_map: Dict[str, List[str]],
) -> Tuple[Dict[str, float], List[Dict[str, Any]], List[str]]:
    per_patient: List[Dict[str, Any]] = []
    missing_predictions: List[str] = []
    predicted_total = 0
    matched_total = 0
    unique_predicted_trials: Set[str] = set()

    unique_pred_lists = _unique_prediction_lists(pred_map)
    single_prediction: Optional[List[str]] = None
    if len(gt_patients) == 1 and len(unique_pred_lists) == 1:
        single_prediction = unique_pred_lists[0]

    for gt in gt_patients:
        predicted_ids: Optional[List[str]] = None
        for alias in gt.aliases:
            if alias in pred_map:
                predicted_ids = pred_map[alias]
                break
        if predicted_ids is None and single_prediction is not None:
            predicted_ids = single_prediction
        if predicted_ids is None:
            missing_predictions.append(gt.patient_id)
            continue

        matched_ids = [tid for tid in predicted_ids if tid in gt.relevant_trial_ids]
        predicted_total += len(predicted_ids)
        matched_total += len(matched_ids)
        unique_predicted_trials.update(predicted_ids)
        per_patient.append(
            {
                "patient_id": gt.patient_id,
                "predicted_count_above_threshold": len(predicted_ids),
                "matched_predicted_count_above_threshold": len(matched_ids),
                "matched_predicted_trials_above_threshold": matched_ids,
            }
        )

    summary = {
        "predicted_count_above_threshold": float(predicted_total),
        "matched_predicted_above_threshold": float(matched_total),
        "unique_predicted_above_threshold": float(len(unique_predicted_trials)),
        "matched_rate_above_threshold": (
            float(matched_total) / float(predicted_total) if predicted_total else 0.0
        ),
    }
    return summary, per_patient, missing_predictions


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ranked_trials.json with Table-style metrics at k=3,5: "
            "Overall Recall@k, MRR@k, Mean Average Rank@k"
        )
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        help="Path to ground truth JSON file or directory",
    )
    parser.add_argument(
        "--predictions-dir",
        default="results",
        help=(
            "Parent directory of ranked_trials.json (single patient) or a root "
            "directory that contains multiple ranked_trials.json files (default: results)"
        ),
    )
    parser.add_argument(
        "--prediction-file",
        default="ranked_trials.json",
        help="Prediction filename to search recursively (default: ranked_trials.json)",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional output JSON path for full report",
    )
    parser.add_argument(
        "--min-pred-score",
        type=float,
        default=None,
        help="Optionally count how many ranked trials have Score >= this threshold.",
    )
    args = parser.parse_args()

    gt_source = Path(args.ground_truth)
    pred_source = Path(args.predictions_dir)

    gt_patients = load_ground_truth(gt_source)
    pred_map = load_predictions(pred_source, args.prediction_file)

    summary, per_patient, missing_predictions = evaluate(gt_patients, pred_map)

    if args.min_pred_score is not None:
        pred_map_above = load_predictions_above_score(
            pred_source, args.prediction_file, args.min_pred_score
        )
        threshold_summary, threshold_per_patient, threshold_missing = (
            compute_threshold_match_stats(gt_patients, pred_map_above)
        )
        summary["min_pred_score"] = float(args.min_pred_score)
        summary.update(threshold_summary)
        if not missing_predictions:
            missing_predictions = threshold_missing
        for item in per_patient:
            for threshold_item in threshold_per_patient:
                if threshold_item["patient_id"] == item["patient_id"]:
                    item.update(threshold_item)
                    break

    report = {
        "summary": summary,
        "missing_predictions": missing_predictions,
        "per_patient": per_patient,
    }

    print("Evaluation Summary")
    print("------------------")
    print(f"Ground-truth patients: {int(summary['ground_truth_patients'])}")
    print(f"Evaluated patients:    {int(summary['evaluated_patients'])}")
    print(
        f"GT matched trials (sum over patients): {int(summary['ground_truth_relevant_total'])}"
    )
    print(
        f"Predicted trials (sum of list lengths): {int(summary['predicted_ranked_total'])}"
    )
    if args.min_pred_score is not None:
        print(
            f"Matched predicted trials: {int(summary['matched_predicted_above_threshold'])}"
        )
        print(
            f"Predicted trials with Score>={args.min_pred_score:g}: "
            f"{int(summary['predicted_count_above_threshold'])}"
        )
        print(
            f"Matched trials with Score>={args.min_pred_score:g}: "
            f"{int(summary['matched_predicted_above_threshold'])}"
        )
        print(
            f"Match rate with Score>={args.min_pred_score:g}: "
            f"{summary['matched_rate_above_threshold']:.6f}"
        )
    else:
        print(
            f"Matched predicted trials: {int(summary['matched_predicted_total'])}"
        )
    print(
        f"GT unique trial IDs:   {int(summary['unique_ground_truth_trials'])}"
    )
    print(
        f"Predicted unique IDs:  {int(summary['unique_predicted_trials'])}"
    )
    print(f"Overall Recall@3:     {summary['overall_recall@3']:.6f}")
    print(f"Overall Recall@5:     {summary['overall_recall@5']:.6f}")
    print(f"MRR@3:                {summary['mrr@3']:.6f}")
    print(f"MRR@5:                {summary['mrr@5']:.6f}")
    print(f"Mean Average Rank@3:  {summary['mean_average_rank@3']:.6f}")
    print(f"Mean Average Rank@5:  {summary['mean_average_rank@5']:.6f}")

    if missing_predictions:
        print(f"Missing predictions:  {len(missing_predictions)}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Saved report to:      {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

