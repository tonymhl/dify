import argparse
import csv
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm


@dataclass
class DifyConfig:
    endpoint: str
    token: str
    top_k: int = 5
    timeout: int = 60
    retries: int = 2
    retry_interval: float = 1.5


def create_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }
    )
    return s


def dify_hit_test(
    session: requests.Session,
    cfg: DifyConfig,
    query: str,
    search_method: str = "semantic_search",
    reranking_enable: bool = False,
) -> Dict[str, Any]:
    payload = {
        "query": query,
        "retrieval_model": {
            "search_method": search_method,
            "reranking_enable": reranking_enable,
            "reranking_mode": None,
            "reranking_model": {
                "reranking_provider_name": "",
                "reranking_model_name": "",
            },
            "weights": None,
            "top_k": cfg.top_k,
            "score_threshold_enabled": False,
            "score_threshold": None,
        },
    }

    last_err: Optional[Exception] = None
    for _ in range(cfg.retries + 1):
        try:
            resp = session.post(cfg.endpoint, json=payload, timeout=cfg.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            time.sleep(cfg.retry_interval)
    raise RuntimeError(f"hit-testing failed after retries: {last_err}")


def extract_top_records(resp_json: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    records = (resp_json or {}).get("records") or []
    return records[:top_k]


def is_hit_in_top(
    records: List[Dict[str, Any]], expected_doc_name: str
) -> Tuple[bool, Optional[int], List[str]]:
    """
    命中标准：topK 的 segment.document.name 中是否包含 CSV 的“文档名”。
    返回：(是否命中, 命中的排名(1-based), 预测文档名列表)
    """
    expected_norm = (expected_doc_name or "").strip().lower()
    predicted_names: List[str] = []
    hit_rank: Optional[int] = None

    for idx, rec in enumerate(records, start=1):
        seg = rec.get("segment") or {}
        doc = seg.get("document") or {}
        doc_name = str(doc.get("name") or "")
        predicted_names.append(doc_name)
        if expected_norm and expected_norm in doc_name.lower():
            if hit_rank is None:
                hit_rank = idx

    return hit_rank is not None, hit_rank, predicted_names


def evaluate_csv(
    cfg: DifyConfig,
    csv_path: str,
    out_result_path: Optional[str] = None,
) -> Dict[str, float]:
    session = create_session(cfg.token)

    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = 0
    hits = 0
    top1 = 0

    if out_result_path:
        out_f = open(out_result_path, "w", encoding="utf-8")
    else:
        out_f = None

    try:
        data_row_idx = 0
        for row in tqdm(rows, desc="Dify评测"):
            query = row.get("问题") or row.get("query") or ""
            expected = row.get("文档名") or row.get("expected") or ""
            if not query or not expected:
                continue
            data_row_idx += 1
            total += 1

            # 读取或推导标签：优先用CSV中的“标签”，否则按前74行EHS，之后“浦江管理制度”
            label = (row.get("标签") or row.get("label") or "").strip()
            if not label:
                label = "EHS" if data_row_idx <= 74 else "浦江管理制度"

            resp_json = dify_hit_test(session, cfg, query=query)
            top_records = extract_top_records(resp_json, cfg.top_k)
            hit, hit_rank, pred_names = is_hit_in_top(
                top_records, expected_doc_name=expected
            )

            if hit:
                hits += 1
                if hit_rank == 1:
                    top1 += 1

            if out_f is not None:
                record = {
                    "query": query,
                    "expected": expected,
                    "label": label,
                    "hit": hit,
                    "hit_rank": hit_rank,
                    "pred_doc_names": pred_names,
                    "raw_count": len(top_records),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if out_f is not None:
            out_f.close()

    metrics = {
        "count": float(total),
        "hit@K": float(hits) / float(total) if total else 0.0,
        "top1": float(top1) / float(total) if total else 0.0,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dify hit-testing 测试与评测（Hit@K，默认K=5）"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="https://dify-aios.gds-services.com/console/api/datasets/820d67aa-6ed6-45e0-95fa-e59285e1397c/hit-testing",
        help="Dify hit-testing 接口完整URL，例如：https://<host>/console/api/datasets/<dataset_id>/hit-testing",
    )
    parser.add_argument(
        "--token",
        type=str,
        default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNmI2NjkyOTktZmVlYi00ZTM1LWJmZTMtOWVkNzY4ZjlmNTRlIiwiZXhwIjoxNzYzNTQ4OTE4LCJpc3MiOiJTRUxGX0hPU1RFRCIsInN1YiI6IkNvbnNvbGUgQVBJIFBhc3Nwb3J0In0.nO0S24vr5C6f3CRu_eCPurN1x3YRd0Q7vyzyLehbiDE",
        help="Dify Bearer Token",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="test_case.csv",
        help="测试集合CSV（列名：问题,文档名）",
    )
    parser.add_argument("--top-k", type=int, default=5, help="评测时请求的TopK")
    parser.add_argument(
        "--out",
        type=str,
        default="output/dify_eval_results.jsonl",
        help="逐条结果（JSONL）输出路径",
    )
    args = parser.parse_args()

    cfg = DifyConfig(
        endpoint=args.endpoint,
        token=args.token,
        top_k=args.top_k,
    )
    metrics = evaluate_csv(cfg=cfg, csv_path=args.csv, out_result_path=args.out)
    print(
        f"评测完成：count={int(metrics['count'])}, hit@{args.top_k}={metrics['hit@K']:.4f}, top1={metrics['top1']:.4f}"
    )


if __name__ == "__main__":
    main()
