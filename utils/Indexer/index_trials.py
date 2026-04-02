#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text())


def make_es_client(cfg: dict) -> Elasticsearch:
    es_conf = cfg["elasticsearch"]
    return Elasticsearch(
        hosts=es_conf["hosts"],
        basic_auth=(es_conf["username"], es_conf["password"]),
        ca_certs=es_conf["ca_certs"],
        verify_certs=True,
    )


def detect_vector_dim(sample: dict) -> int:
    for k, v in sample.items():
        if k.endswith("_vector") and isinstance(v, list):
            return len(v)
    raise ValueError("No vector field found in sample")


def load_processed(folder: Path) -> list[dict]:
    docs = []
    for fn in os.listdir(folder):
        if fn.endswith(".json"):
            docs.append(json.loads((folder / fn).read_text()))
    return docs


def _is_zero_vector(vec: list) -> bool:
    return isinstance(vec, list) and len(vec) > 0 and all(v == 0.0 for v in vec)


def clean_doc_for_es(doc: dict) -> dict:
    """Remove None/null values and zero-magnitude vectors that violate the ES mapping."""
    cleaned = {}
    for key, value in doc.items():
        if value is None:
            continue
        if key in ("start_date", "completion_date") and not value:
            continue
        if key in ("minimum_age", "maximum_age"):
            try:
                cleaned[key] = float(value)
            except (ValueError, TypeError):
                continue
        elif key.endswith("_vector"):
            if _is_zero_vector(value):
                continue
            cleaned[key] = value
        else:
            cleaned[key] = value
    return cleaned


def create_index(es: Elasticsearch, name: str, dims: int):
    body = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "standard_lowercase": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase"],
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "nct_id": {"type": "keyword"},
                "brief_title": {"type": "text", "analyzer": "standard_lowercase"},
                "brief_title_vector": {"type": "dense_vector", "dims": dims},
                "brief_summary": {"type": "text", "analyzer": "standard_lowercase"},
                "brief_summary_vector": {"type": "dense_vector", "dims": dims},
                "condition": {"type": "text", "analyzer": "standard_lowercase"},
                "condition_vector": {"type": "dense_vector", "dims": dims},
                "overall_status": {"type": "keyword"},
                "start_date": {"type": "date", "format": "yyyy-MM-dd"},
                "completion_date": {"type": "date", "format": "yyyy-MM-dd"},
                "phase": {"type": "keyword"},
                "study_type": {"type": "keyword"},
                "intervention": {
                    "properties": {
                        "intervention_type": {"type": "keyword"},
                        "intervention_name": {"type": "text"},
                    }
                },
                "gender": {"type": "keyword"},
                "minimum_age": {"type": "float"},
                "maximum_age": {"type": "float"},
                "location": {
                    "properties": {
                        "location_name": {"type": "text"},
                        "location_address": {"type": "text"},
                    }
                },
                "reference": {
                    "type": "nested",
                    "properties": {
                        "citation": {"type": "text"},
                        "PMID": {"type": "keyword"},
                    },
                },
                "eligibility_criteria": {
                    "type": "text",
                    "analyzer": "standard_lowercase",
                },
                "eligibility_criteria_vector": {"type": "dense_vector", "dims": dims},
            }
        },
    }
    es.indices.create(index=name, body=body)
    print(f"Created index `{name}` with vector dims={dims}")


def main():
    parser = argparse.ArgumentParser(description="Bulk‑index processed trial JSONs")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file with Elasticsearch credentials",
    )
    parser.add_argument(
        "--processed-folder", required=True, help="Folder of processed JSONs to index"
    )
    parser.add_argument(
        "--index-name",
        default="clinical_trials",
        help="Target Elasticsearch index name",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Number of docs per bulk request"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    es = make_es_client(cfg)

    processed_path = Path(args.processed_folder)
    docs = load_processed(processed_path)
    if not docs:
        print("❌ No JSONs found to index.")
        return

    dims = detect_vector_dim(docs[0])

    # <-- FIXED: use keyword arg `index=`
    if not es.indices.exists(index=args.index_name):
        create_index(es, args.index_name, dims)
    else:
        print(f"Index `{args.index_name}` already exists; skipping creation.")

    actions = [
        {
            "_op_type": "index",
            "_index": args.index_name,
            "_id": doc["nct_id"],
            "_source": clean_doc_for_es(doc),
        }
        for doc in docs
    ]

    success, errors = bulk(
        client=es,
        actions=actions,
        chunk_size=args.batch_size,
        stats_only=False,
        raise_on_error=False,
    )
    es.indices.refresh(index=args.index_name)
    if errors:
        print(f"\n⚠️  {len(errors)} failures. First 3 errors:")
        for err in errors[:3]:
            print(json.dumps(err, indent=2, default=str))
    print(f"\n✅ Indexed {success} documents; {len(errors)} failures.")


if __name__ == "__main__":
    main()


