#!/usr/bin/env python3
import argparse
import json
import re
import socket
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_BASE_URL = "http://127.0.0.1:9220/v1"
DEFAULT_API_KEY = "321"
DEFAULT_MODEL = "h200-qwen3.5-122b"
RETRYABLE_HTTP_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def list_json_files(path: Path) -> List[Path]:
    if path.is_file():
        if path.suffix.lower() != ".json":
            raise ValueError(f"Expected a .json file, got: {path}")
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if not path.is_dir():
        raise ValueError(f"Path must be a JSON file or directory: {path}")
    return sorted([p for p in path.iterdir() if p.suffix.lower() == ".json"])


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _join_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, list):
        parts = [_join_text(v) for v in value]
        return normalize_text(" ".join([p for p in parts if p]))
    if isinstance(value, dict):
        parts = []
        for key in ("label", "description", "id", "text", "name"):
            if value.get(key):
                parts.append(str(value[key]))
        if parts:
            return normalize_text(" ".join(parts))
    return normalize_text(str(value))


def patient_id_from_path(path: Path, data: Dict[str, Any]) -> str:
    return str(data.get("id") or data.get("patient_id") or path.stem)


def build_patient_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    main_conditions = data.get("main_conditions") or []
    other_conditions = data.get("other_conditions") or []
    expanded_sentences = data.get("expanded_sentences") or []

    # Processed patient file / keywords.json-like structure.
    if main_conditions or other_conditions or expanded_sentences:
        summary_lines = []
        if main_conditions:
            summary_lines.append("主要疾病/问题: " + "；".join(map(str, main_conditions[:15])))
        if other_conditions:
            summary_lines.append("其他相关情况: " + "；".join(map(str, other_conditions[:20])))
        if expanded_sentences:
            summary_lines.append("患者描述: " + " ".join(map(str, expanded_sentences[:20])))
        return {
            "main_conditions": [str(x) for x in main_conditions],
            "other_conditions": [str(x) for x in other_conditions],
            "summary_text": "\n".join(summary_lines),
        }

    # Raw phenopacket-like structure.
    subject = data.get("subject") or {}
    phenotypes = []
    for pf in data.get("phenotypicFeatures", []) or []:
        pf_type = pf.get("type") or {}
        label = pf_type.get("label") or pf_type.get("id")
        desc = pf.get("description")
        if label and desc:
            phenotypes.append(f"{label}: {desc}")
        elif label:
            phenotypes.append(str(label))

    diseases = []
    for disease in data.get("diseases", []) or []:
        term = disease.get("term") or {}
        label = term.get("label") or term.get("id")
        if label:
            diseases.append(str(label))

    lines = []
    if subject.get("description"):
        lines.append(f"患者描述: {subject['description']}")
    if diseases:
        lines.append("疾病: " + "；".join(diseases[:15]))
    if phenotypes:
        lines.append("表型/症状: " + "；".join(phenotypes[:20]))

    return {
        "main_conditions": diseases[:10],
        "other_conditions": phenotypes[:20],
        "summary_text": "\n".join(lines) or json.dumps(data, ensure_ascii=False),
    }


def build_trial_summary(data: Dict[str, Any]) -> str:
    lines = []
    for key, label in [
        ("nct_id", "试验ID"),
        ("brief_title", "标题"),
        ("official_title", "正式标题"),
        ("overall_status", "状态"),
        ("phase", "分期"),
        ("condition", "疾病"),
        ("brief_summary", "摘要"),
        ("detailed_description", "详细描述"),
        ("eligibility_criteria", "入排标准"),
        ("gender", "性别要求"),
        ("minimum_age", "最小年龄"),
        ("maximum_age", "最大年龄"),
    ]:
        text = _join_text(data.get(key))
        if text:
            lines.append(f"{label}: {text}")
    return "\n".join(lines)


def lexical_prefilter_score(patient: Dict[str, Any], trial: Dict[str, Any]) -> float:
    trial_blob = " ".join(
        [
            _join_text(trial.get("condition")),
            _join_text(trial.get("brief_title")),
            _join_text(trial.get("official_title")),
            _join_text(trial.get("brief_summary")),
            _join_text(trial.get("eligibility_criteria")),
        ]
    ).lower()
    if not trial_blob:
        return 0.0

    score = 0.0
    for term in patient.get("main_conditions", []):
        term_l = str(term).strip().lower()
        if term_l and term_l in trial_blob:
            score += 3.0
    for term in patient.get("other_conditions", []):
        term_l = str(term).strip().lower()
        if term_l and term_l in trial_blob:
            score += 1.0
    return score


def extract_json_block(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def build_trial_criteria_for_cot(trial_data: Dict[str, Any]) -> str:
    """Same criteria block shape as Matcher.pipeline.cot_reasoning.BatchTrialProcessor."""
    crit = _join_text(trial_data.get("eligibility_criteria"))
    if crit:
        return f"Eligibility Criteria:\n{crit}"
    return "No eligibility criteria provided."


def format_cot_eligibility_prompts(
    criteria_text_formatted: str, patient_profile: str
) -> tuple[str, str]:
    """Mirror cot_reasoning.py use_cot=True system + user messages (English)."""
    system_msg = (
        "You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. "
        "Answer the following question. Before answering, create a concise chain of thoughts reasoning to ensure a logical and accurate response.\n"
    )
    user_msg = (
        "Assess the given patient's eligibility for a clinical trial by evaluating each and every criterion individually.\n\n"
        "### INCLUSION CRITERIA ASSESSMENT\n"
        "For each inclusion criterion, classify it as one of:\n"
        "- **Met:** The patient's data explicitly and unequivocally satisfies the criterion.\n"
        "- **Not Met:** The patient's data explicitly and unequivocally contradicts or fails to satisfy the criterion.\n"
        "- **Unclear:** Insufficient or missing patient data to verify.\n"
        "- **Irrelevant:** The criterion does not apply to the patient's context.\n\n"
        "### EXCLUSION CRITERIA ASSESSMENT\n"
        "For each exclusion criterion, classify it as one of:\n"
        "- **Violated:** The patient's data explicitly and unequivocally violates the criterion.\n"
        "- **Not Violated:** The patient's data confirms compliance with the criterion.\n"
        "- **Unclear:** Insufficient or missing patient data to verify.\n"
        "- **Irrelevant:** The criterion does not apply to the patient's context.\n\n"
        "### IMPORTANT INSTRUCTIONS\n"
        "- Ensure all criteria are assessed one-by-one.\n"
        "- Use **only** the provided patient data; **do not infer, assume, or extrapolate beyond the given information.**\n"
        "- Justifications must be strictly based on direct evidence from the patient profile.\n"
        "- Return exactly one JSON object and no extra prose, no markdown fences, and no leading or trailing commentary.\n"
        "### RESPONSE FORMAT (STRICTLY FOLLOW)\n"
        "{\n"
        '  "Inclusion_Criteria_Evaluation": [\n'
        '    {"Criterion": "Exact inclusion criterion text", "Classification": "Met | Not Met | Unclear | Irrelevant", "Justification": "Clear, evidence-based rationale using ONLY provided data"}\n'
        "  ],\n"
        '  "Exclusion_Criteria_Evaluation": [\n'
        '    {"Criterion": "Exact exclusion criterion text", "Classification": "Violated | Not Violated | Unclear | Irrelevant", "Justification": "Clear, evidence-based rationale using ONLY provided data"}\n'
        "  ],\n"
        '  "Recap": "Concise summary of key qualifying/disqualifying factors",\n'
        '  "Final Decision": "Eligible | Likely Eligible (leaning toward inclusion) | Likely Ineligible (leaning toward exclusion) | Ineligible"\n'
        "}\n\n"
        "### INPUT\n"
        "---Start of Clinical Trial Criteria---\n"
        f"{criteria_text_formatted}\n"
        "---End of Clinical Trial Criteria---\n\n"
        "----\n"
        "---Start of Patient Description---\n"
        f"{patient_profile}\n"
        "Written informed consent has been obtained from the patient or their legal representative.\n"
        "---End of Patient Description---\n"
        "## IMPORTANT REMINDER:\n"
        "NEVER make assumptions, inferences, or extrapolations beyond the explicitly stated patient information."
    )
    return system_msg, user_msg


def _final_decision_to_match(final_decision: str) -> tuple[bool, float]:
    s = (final_decision or "").strip().lower()
    if "ineligible" in s and "likely" not in s:
        return False, 0.15
    if s.startswith("ineligible") or (
        "likely ineligible" in s or "leaning toward exclusion" in s
    ):
        return False, 0.35
    if "likely eligible" in s or "leaning toward inclusion" in s:
        return True, 0.72
    if s.startswith("eligible") or "eligible" == s:
        return True, 0.92
    return False, 0.4


def map_cot_eligibility_json_to_match_assessment(
    raw: Dict[str, Any],
) -> Dict[str, Any]:
    """Map CoT eligibility JSON (cot_reasoning schema) to this script's assessment shape."""
    recap = raw.get("Recap") or raw.get("recap") or ""
    final = (
        raw.get("Final Decision")
        or raw.get("Final_Decision")
        or raw.get("final_decision")
        or ""
    )
    matched, score = _final_decision_to_match(str(final))

    inc = raw.get("Inclusion_Criteria_Evaluation") or raw.get(
        "inclusion_criteria_evaluation"
    ) or []
    exc = raw.get("Exclusion_Criteria_Evaluation") or raw.get(
        "exclusion_criteria_evaluation"
    ) or []

    missing: List[str] = []
    conflicts: List[str] = []

    for item in inc:
        if not isinstance(item, dict):
            continue
        cls = str(item.get("Classification") or item.get("classification") or "")
        cls_l = cls.lower()
        crit = item.get("Criterion") or item.get("criterion")
        crit_s = str(crit).strip() if crit else ""
        if "unclear" in cls_l and crit_s:
            missing.append(crit_s)
        if "not met" in cls_l and crit_s:
            conflicts.append(f"Inclusion not met: {crit_s}")

    for item in exc:
        if not isinstance(item, dict):
            continue
        cls = str(item.get("Classification") or item.get("classification") or "")
        cls_l = cls.lower()
        crit = item.get("Criterion") or item.get("criterion")
        crit_s = str(crit).strip() if crit else ""
        if "unclear" in cls_l and crit_s:
            missing.append(crit_s)
        if "violated" in cls_l and crit_s:
            conflicts.append(f"Exclusion violated: {crit_s}")

    reason = str(recap).strip() if recap else str(final).strip()
    return {
        "matched": matched,
        "match_score": score,
        "reason": reason or "CoT eligibility assessment",
        "missing_information": missing,
        "conflicts": conflicts,
        "use_cot_reasoning": True,
        "final_decision": str(final).strip(),
        "cot_eligibility": raw,
    }


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = 180,
        provider_timeout_seconds: int = 0,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.provider_timeout_seconds = provider_timeout_seconds

    def _post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _with_timeout_hints(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Attach provider timeout hints for gateways that support them.

        Note: support depends on the upstream provider. Unknown fields may be
        ignored by many OpenAI-compatible gateways.
        """
        if self.provider_timeout_seconds <= 0:
            return payload
        enhanced = dict(payload)
        # Common field names seen across OpenAI-compatible providers.
        enhanced["timeout"] = self.provider_timeout_seconds
        enhanced["request_timeout"] = self.provider_timeout_seconds
        return enhanced

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: Optional[int] = None,
        completions_suffix: str = "请只输出 JSON 对象，不要输出解释。",
    ) -> Dict[str, Any]:
        chat_url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload = self._with_timeout_hints(payload)
        try:
            data = self._post(chat_url, payload)
            content = data["choices"][0]["message"]["content"]
            return extract_json_block(content)
        except urllib.error.HTTPError as exc:
            # Fallback for providers that only expose `/completions`.
            if exc.code not in (400, 404, 405):
                raise

        completion_url = f"{self.base_url}/completions"
        prompt = (
            f"{system_prompt}\n\n"
            f"{user_prompt}\n\n"
            f"{completions_suffix}"
        )
        comp_max = max_tokens if max_tokens is not None else 1200
        payload = {
            "model": self.model,
            "temperature": 0,
            "prompt": prompt,
            "max_tokens": comp_max,
        }
        payload = self._with_timeout_hints(payload)
        data = self._post(completion_url, payload)
        text = data["choices"][0].get("text", "")
        return extract_json_block(text)


def is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in RETRYABLE_HTTP_CODES
    if isinstance(exc, urllib.error.URLError):
        return True
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, socket.timeout):
        return True
    return False


def format_error(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        detail = f"{type(exc).__name__} HTTP {exc.code}: {exc.reason} url={exc.url}"
        try:
            body = exc.read().decode("utf-8", errors="replace").strip()
            if body:
                detail += f" body={body[:500]}"
        except Exception:
            pass
        return detail
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", None)
        if reason is not None:
            return f"{type(exc).__name__}: {exc} (reason={type(reason).__name__}: {reason})"
    return f"{type(exc).__name__}: {exc}"


def judge_trial_match_with_retry(
    client: OpenAICompatClient,
    patient_id: str,
    patient_summary: Dict[str, Any],
    trial_data: Dict[str, Any],
    *,
    max_retries: int,
    retry_backoff_seconds: float,
    retry_max_backoff_seconds: float,
    verbose_errors: bool,
    use_cot_reasoning: bool = False,
    cot_max_tokens: int = 4000,
) -> Dict[str, Any]:
    trial_id = str(trial_data.get("nct_id") or trial_data.get("trial_id") or "")
    attempts = max_retries + 1
    last_exc: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        try:
            return judge_trial_match(
                client,
                patient_id,
                patient_summary,
                trial_data,
                use_cot_reasoning=use_cot_reasoning,
                cot_max_tokens=cot_max_tokens,
            )
        except Exception as exc:
            last_exc = exc
            if not is_retryable_error(exc) or attempt >= attempts:
                break

            sleep_for = retry_backoff_seconds * (2 ** (attempt - 1))
            if retry_max_backoff_seconds > 0:
                sleep_for = min(sleep_for, retry_max_backoff_seconds)
            print(
                f"  [{patient_id}] trial {trial_id} request failed "
                f"(attempt {attempt}/{attempts}): {format_error(exc)}; "
                f"retrying in {sleep_for:.1f}s"
            )
            if verbose_errors:
                tb = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ).rstrip()
                if tb:
                    print(f"    traceback:\n{tb}")
            time.sleep(sleep_for)

    assert last_exc is not None
    raise last_exc


def judge_trial_match(
    client: OpenAICompatClient,
    patient_id: str,
    patient_summary: Dict[str, Any],
    trial_data: Dict[str, Any],
    *,
    use_cot_reasoning: bool = False,
    cot_max_tokens: int = 4000,
) -> Dict[str, Any]:
    trial_id = str(trial_data.get("nct_id") or trial_data.get("trial_id") or "")

    if use_cot_reasoning:
        criteria_fmt = build_trial_criteria_for_cot(trial_data)
        system_prompt, user_prompt = format_cot_eligibility_prompts(
            criteria_fmt, patient_summary["summary_text"]
        )
        raw = client.chat_json(
            system_prompt,
            user_prompt,
            max_tokens=cot_max_tokens,
            completions_suffix="Please output only a JSON object.",
        )
        result = map_cot_eligibility_json_to_match_assessment(raw)
        result["trial_id"] = trial_id
        return result

    system_prompt = (
        "你是临床试验匹配助手。"
        "请根据患者信息与试验信息，判断该患者是否可能匹配该临床试验。"
        "重点关注疾病是否相关、关键入排标准是否明显冲突、患者信息是否足以支持“可能匹配”。"
        "输出必须是 JSON。"
    )

    user_prompt = f"""
患者ID: {patient_id}

患者信息:
{patient_summary["summary_text"]}

临床试验信息:
{build_trial_summary(trial_data)}

请返回如下 JSON:
{{
  "matched": true,
  "match_score": 0.0,
  "reason": "简短中文理由",
  "missing_information": ["若无则返回空数组"],
  "conflicts": ["若无则返回空数组"]
}}

要求:
1. matched 只能是 true 或 false。
2. match_score 为 0 到 1 的浮点数。
3. 如果疾病明显不相关，matched 必须为 false。
4. 如果患者信息不足但疾病高度相关且没有明显冲突，可给 true，但 reason 里说明是“可能匹配，待补充信息”。
5. 只输出 JSON，不要输出其他文字。
""".strip()

    result = client.chat_json(system_prompt, user_prompt)
    result["trial_id"] = trial_id
    return result


def _trial_id(trial: Dict[str, Any]) -> str:
    return str(trial.get("nct_id") or trial.get("trial_id") or "")


def _load_existing_progress(output_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load previous assessments keyed by trial_id so we can skip them."""
    if not output_path.exists():
        return {}
    try:
        data = load_json(output_path)
        return {
            a["trial_id"]: a
            for a in data.get("all_assessments", [])
            if a.get("trial_id")
        }
    except Exception:
        return {}


def _save_patient_result(
    output_path: Path,
    pid: str,
    patient_path: Path,
    total_candidates: int,
    assessments: List[Dict[str, Any]],
    min_match_score: float,
    *,
    complete: bool,
) -> None:
    matched_trials = [
        item
        for item in assessments
        if bool(item.get("matched"))
        and float(item.get("match_score", 0.0)) >= min_match_score
    ]
    matched_trials.sort(
        key=lambda x: float(x.get("match_score", 0.0)), reverse=True
    )
    output = {
        "patient_id": pid,
        "patient_file": str(patient_path),
        "trials_evaluated": total_candidates,
        "trials_assessed": len(assessments),
        "complete": complete,
        "matched_trial_count": len(matched_trials),
        "matched_trials": matched_trials,
        "all_assessments": assessments,
    }
    dump_json(output_path, output)


def _build_failed_assessment(trial_id: str, error_message: str) -> Dict[str, Any]:
    return {
        "trial_id": trial_id,
        "matched": False,
        "match_score": 0.0,
        "reason": f"API request failed after retries: {error_message}",
        "missing_information": [],
        "conflicts": [f"api_error: {error_message}"],
        "api_error": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Use Qwen 122B via OpenAI-compatible API to match patients to trials."
    )
    parser.add_argument(
        "--trials-dir",
        default="data/custom/processed_cancer_trials",
        help="Directory containing trial JSON files, or a single trial JSON file.",
    )
    parser.add_argument(
        "--patients-dir",
        default="example/processed_patients",
        help="Directory containing patient JSON files, or a single patient JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/qwen_patient_trial_matches",
        help="Directory to write per-patient match results.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible base URL. Default: http://127.0.0.1:9220/v1",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="API key for the OpenAI-compatible endpoint. Default: 321",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model ID to call.",
    )
    parser.add_argument(
        "--prefilter-top-k",
        type=int,
        default=0,
        help="If > 0, use lexical prefilter and only send top-k trials per patient to the LLM. 0 means evaluate all trials.",
    )
    parser.add_argument(
        "--min-match-score",
        type=float,
        default=0.5,
        help="Only keep matched trials with score >= this value in matched_trials.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between API calls to avoid rate limits.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save progress every N trial assessments (default: 5).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore previous progress and re-evaluate everything from scratch.",
    )
    parser.add_argument(
        "--timeout",
        "--http-timeout",
        type=int,
        dest="timeout",
        default=180,
        help="HTTP client timeout in seconds for each API request (default: 180).",
    )
    parser.add_argument(
        "--provider-timeout-seconds",
        type=int,
        default=0,
        help=(
            "If > 0, include timeout hints in request payload (timeout/request_timeout) "
            "for compatible OpenAI-style gateways."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Retry count for transient API failures like 504/502/503 (default: 4).",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=5.0,
        help="Base backoff in seconds between retries; doubles each retry (default: 5.0).",
    )
    parser.add_argument(
        "--retry-max-backoff-seconds",
        type=float,
        default=60.0,
        help="Maximum backoff wait per retry in seconds (default: 60.0; <=0 means no cap).",
    )
    parser.add_argument(
        "--skip-failed-trials",
        action="store_true",
        help="If a trial still fails after retries, record it as an API failure and continue.",
    )
    parser.add_argument(
        "--verbose-errors",
        action="store_true",
        help="Print detailed traceback when API requests fail and retry.",
    )
    parser.add_argument(
        "--use-cot-reasoning",
        action="store_true",
        help=(
            "Use English chain-of-thought eligibility prompts aligned with "
            "source/Matcher/pipeline/cot_reasoning.py (trial eligibility_criteria + JSON schema)."
        ),
    )
    parser.add_argument(
        "--cot-max-tokens",
        type=int,
        default=4000,
        help="max_tokens for chat/completions when --use-cot-reasoning is set (default: 4000).",
    )
    args = parser.parse_args()

    trials_dir = Path(args.trials_dir)
    patients_dir = Path(args.patients_dir)
    output_dir = Path(args.output_dir)
    base_url, api_key, model_id = (
        args.base_url.rstrip("/"),
        args.api_key,
        args.model,
    )
    client = OpenAICompatClient(
        base_url=base_url,
        api_key=api_key,
        model=model_id,
        timeout=args.timeout,
        provider_timeout_seconds=args.provider_timeout_seconds,
    )

    trial_files = list_json_files(trials_dir)
    patient_files = list_json_files(patients_dir)
    if not trial_files:
        raise FileNotFoundError(f"No trial JSON files found in {trials_dir}")
    if not patient_files:
        raise FileNotFoundError(f"No patient JSON files found in {patients_dir}")

    trials = [load_json(p) for p in trial_files]
    print(f"Loaded {len(trials)} trials and {len(patient_files)} patients.")

    total_skipped_patients = 0
    total_skipped_trials = 0
    total_new_assessments = 0

    for pat_idx, patient_path in enumerate(patient_files, start=1):
        patient_data = load_json(patient_path)
        pid = patient_id_from_path(patient_path, patient_data)
        patient_summary = build_patient_summary(patient_data)
        output_path = output_dir / f"{pid}.json"

        candidate_trials = trials
        if args.prefilter_top_k > 0:
            scored = [
                (lexical_prefilter_score(patient_summary, trial), trial)
                for trial in trials
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            candidate_trials = [trial for score, trial in scored[: args.prefilter_top_k] if score > 0]
            if not candidate_trials:
                candidate_trials = [trial for _, trial in scored[: args.prefilter_top_k]]

        # --- Resume logic ---
        existing: Dict[str, Dict[str, Any]] = {}
        if not args.no_resume:
            existing = _load_existing_progress(output_path)

        # Check if this patient is already fully processed
        if existing and len(existing) >= len(candidate_trials):
            all_covered = all(
                _trial_id(t) in existing for t in candidate_trials
            )
            if all_covered:
                total_skipped_patients += 1
                print(
                    f"[{pat_idx}/{len(patient_files)}] [{pid}] already complete "
                    f"({len(existing)} trials) — skipped"
                )
                continue

        # Separate pending vs already-done trials
        pending_trials = []
        for trial in candidate_trials:
            tid = _trial_id(trial)
            if tid not in existing:
                pending_trials.append(trial)

        resumed_count = len(existing)
        assessments = list(existing.values())
        total_skipped_trials += resumed_count

        if resumed_count > 0:
            print(
                f"[{pat_idx}/{len(patient_files)}] [{pid}] resuming — "
                f"{resumed_count} done, {len(pending_trials)} remaining"
            )
        else:
            print(
                f"[{pat_idx}/{len(patient_files)}] [{pid}] "
                f"evaluating {len(pending_trials)} trial(s)..."
            )

        new_count = 0
        for idx, trial in enumerate(pending_trials, start=1):
            tid = _trial_id(trial)
            try:
                result = judge_trial_match_with_retry(
                    client,
                    pid,
                    patient_summary,
                    trial,
                    max_retries=args.max_retries,
                    retry_backoff_seconds=args.retry_backoff_seconds,
                    retry_max_backoff_seconds=args.retry_max_backoff_seconds,
                    verbose_errors=args.verbose_errors,
                    use_cot_reasoning=args.use_cot_reasoning,
                    cot_max_tokens=args.cot_max_tokens,
                )
                assessments.append(result)
            except Exception as exc:
                error_message = format_error(exc)
                if args.skip_failed_trials:
                    print(
                        f"  [{pid}] trial {tid} failed after retries; "
                        f"recording failure and continuing: {error_message}"
                    )
                    assessments.append(
                        _build_failed_assessment(tid, error_message)
                    )
                    new_count += 1
                    total_new_assessments += 1
                    if new_count % args.save_every == 0:
                        _save_patient_result(
                            output_path,
                            pid,
                            patient_path,
                            len(candidate_trials),
                            assessments,
                            args.min_match_score,
                            complete=False,
                        )
                    continue
                _save_patient_result(
                    output_path,
                    pid,
                    patient_path,
                    len(candidate_trials),
                    assessments,
                    args.min_match_score,
                    complete=False,
                )
                raise RuntimeError(
                    f"API evaluation failed for patient {pid}, trial {tid}: {error_message}"
                ) from exc
            new_count += 1
            total_new_assessments += 1

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

            # Incremental save
            if new_count % args.save_every == 0:
                _save_patient_result(
                    output_path, pid, patient_path,
                    len(candidate_trials), assessments,
                    args.min_match_score, complete=False,
                )

            if idx % 10 == 0 or idx == len(pending_trials):
                done_total = resumed_count + idx
                print(
                    f"  [{pid}] {done_total}/{len(candidate_trials)} "
                    f"(+{idx} new)"
                )

        # Final save for this patient (mark complete)
        _save_patient_result(
            output_path, pid, patient_path,
            len(candidate_trials), assessments,
            args.min_match_score, complete=True,
        )
        matched_count = sum(
            1 for a in assessments
            if bool(a.get("matched"))
            and float(a.get("match_score", 0.0)) >= args.min_match_score
        )
        print(
            f"  [{pid}] complete — {matched_count} matched "
            f"out of {len(candidate_trials)}"
        )

    # Final summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print(f"  Patients total:        {len(patient_files)}")
    print(f"  Patients skipped:      {total_skipped_patients} (already complete)")
    print(f"  Patients processed:    {len(patient_files) - total_skipped_patients}")
    print(f"  Trials resumed:        {total_skipped_trials} (from previous runs)")
    print(f"  Trials newly assessed: {total_new_assessments}")
    print(f"  Output directory:      {output_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

