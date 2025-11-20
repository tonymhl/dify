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
    csrf_token: Optional[str] = None
    # Retrieval configs
    search_method: str = "semantic_search"
    reranking_enable: bool = False
    reranking_mode: str = "weighted_score"
    rerank_provider: str = ""
    rerank_model: str = ""
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    top_k: int = 10
    # Request configs
    timeout: int = 60
    retries: int = 2
    retry_interval: float = 1.5


def create_session(token: str, csrf_token: Optional[str] = None) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
    )
    # Console API 使用 Cookie 中的 access_token 而非 Authorization Header
    # 不指定 domain，让 requests 自动处理
    s.cookies.set("access_token", token)
    
    # 如果提供了 CSRF Token，也加入 Cookie 和 Header
    if csrf_token:
        s.cookies.set("csrf_token", csrf_token)
        s.headers.update({"x-csrf-token": csrf_token})
    
    return s


def dify_hit_test(
    session: requests.Session,
    cfg: DifyConfig,
    query: str,
    debug: bool = False,
) -> Dict[str, Any]:
    # 构造符合 Console API /hit-testing 接口的 retrieval_model 参数
    # 参考 curl 结构：
    # "retrieval_model": {
    #     "search_method": "semantic_search",
    #     "reranking_enable": false,
    #     "reranking_mode": "weighted_score",
    #     "reranking_model": {
    #         "reranking_provider_name": "langgenius/tongyi/tongyi",
    #         "reranking_model_name": "gte-rerank-v2"
    #     },
    #     "weights": {
    #         "weight_type": "customized",
    #         "keyword_setting": { "keyword_weight": 0.3 },
    #         "vector_setting": {
    #             "vector_weight": 0.7,
    #             "embedding_model_name": "",
    #             "embedding_provider_name": ""
    #         }
    #     },
    #     "top_k": 10,
    #     "score_threshold_enabled": false,
    #     "score_threshold": 0.5
    # }

    # 构造 reranking_model 对象
    # 无论是否 enable rerank，Console API 似乎都接受这个结构，只要 reranking_enable=false 即可
    # 这里我们为了完整性，总是构造它
    reranking_model_obj = {
        "reranking_provider_name": cfg.rerank_provider,
        "reranking_model_name": cfg.rerank_model,
    }

    # 构造 weights 对象 (用于 Hybrid Search 或 Weighted Score 模式)
    # 注意：Console API 的 weights 结构比较复杂，包含 vector_setting 和 keyword_setting
    weights_obj = None
    if cfg.search_method == "hybrid_search" or True: # 实际上很多时候即使非 hybrid 也会带上默认权重结构
        weights_obj = {
            "weight_type": "customized",
            "keyword_setting": {
                "keyword_weight": cfg.keyword_weight
            },
            "vector_setting": {
                "vector_weight": cfg.semantic_weight,
                "embedding_model_name": "",  # 通常为空，后端会自动使用 Dataset 绑定的 Embedding
                "embedding_provider_name": ""
            }
        }

    retrieval_model = {
        "search_method": cfg.search_method,
        "reranking_enable": cfg.reranking_enable,
        "reranking_mode": cfg.reranking_mode, # e.g. "weighted_score" or "reranking_model"
        "reranking_model": reranking_model_obj,
        "weights": weights_obj,
        "top_k": cfg.top_k,
        "score_threshold_enabled": False,
        "score_threshold": 0.5, # 默认阈值，可参数化
    }

    payload = {
        "query": query,
        "retrieval_model": retrieval_model,
    }

    if debug:
        print(f"\n[DEBUG] Payload: {json.dumps(payload, ensure_ascii=False)}")

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
    debug: bool = False,
) -> Dict[str, float]:
    session = create_session(cfg.token, cfg.csrf_token)

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

            label = (row.get("标签") or row.get("label") or "").strip()
            if not label:
                label = "EHS" if data_row_idx <= 74 else "浦江管理制度"

            resp_json = dify_hit_test(session, cfg, query=query, debug=(debug and data_row_idx <= 3))
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
        description="Dify hit-testing 测试与评测 (使用 Console API)"
    )
    # 默认 Endpoint 修改为 Console API 的 hit-testing
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost/console/api/datasets/fe5ac293-aec6-4a51-b373-278c8f2cfe09/hit-testing",
        help="Dify Console API Hit-Testing Endpoint",
    )
    # Token 这里是 Console API 的 Access Token (Bearer Token)，不是 Dataset API Key
    parser.add_argument(
        "--token",
        type=str,
        default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiZjMyZmY1NzktNTk5MS00MGE4LWE0NGMtMWQ3ZjZiNjE0NDljIiwiZXhwIjoxNzYzNjMzMTMxLCJpc3MiOiJTRUxGX0hPU1RFRCIsInN1YiI6IkNvbnNvbGUgQVBJIFBhc3Nwb3J0In0.vbaBMBtIvVbR7CZSAoqJMdSTfzjg-mgjniSAMyiI0Ac",
        help="Dify Console API Access Token (Bearer ...)",
    )
    parser.add_argument(
        "--csrf-token",
        type=str,
        default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NjM2MzMxMzEsInN1YiI6ImYzMmZmNTc5LTU5OTEtNDBhOC1hNDRjLTFkN2Y2YjYxNDQ5YyJ9.bQRWdB9ieO2ZDr0XMcLtuDuWhj3LihqLRAm1iRqPSWY",
        help="Dify Console API CSRF Token",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="test_case/test_case.csv",
        help="测试集合CSV（列名：问题,文档名）",
    )
    parser.add_argument("--top-k", type=int, default=10, help="TopK")
    parser.add_argument(
        "--out",
        type=str,
        default="output/dify_eval_results.jsonl",
        help="逐条结果（JSONL）输出路径",
    )
    
    # 检索参数
    parser.add_argument(
        "--search-method",
        type=str,
        default="hybrid_search",
        choices=["hybrid_search", "semantic_search", "full_text_search", "keyword_search"],
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.7,
        help="Semantic Weight (Vector)",
    )
    parser.add_argument(
        "--keyword-weight",
        type=float,
        default=0.3,
        help="Keyword Weight",
    )
    
    # Rerank 参数
    parser.add_argument(
        "--enable-rerank",
        action="store_true",
        help="Enable Rerank",
    )
    parser.add_argument(
        "--rerank-provider",
        type=str,
        default="langgenius/tongyi/tongyi", # 注意：Console API 需要完整的 provider string
        help="Rerank Provider Name",
    )
    parser.add_argument(
        "--rerank-model",
        type=str,
        default="gte-rerank-v2",
        help="Rerank Model Name",
    )
    parser.add_argument(
        "--rerank-mode",
        type=str,
        default="reranking_model", # 或者是 weighted_score，取决于是否启用 rerank
        help="Reranking Mode (e.g., reranking_model, weighted_score)",
    )

    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # 根据是否 enable rerank 自动调整 reranking_mode，除非手动指定
    # 如果开启 rerank -> reranking_model
    # 如果关闭 rerank -> weighted_score (这似乎是 Console API 在混合检索且无关 Rerank 时的默认行为)
    reranking_mode = args.rerank_mode
    if args.enable_rerank:
        reranking_mode = "reranking_model"
    else:
        reranking_mode = "weighted_score"

    cfg = DifyConfig(
        endpoint=args.endpoint,
        token=args.token,
        csrf_token=args.csrf_token,
        top_k=args.top_k,
        search_method=args.search_method,
        reranking_enable=args.enable_rerank,
        reranking_mode=reranking_mode,
        rerank_provider=args.rerank_provider,
        rerank_model=args.rerank_model,
        semantic_weight=args.semantic_weight,
        keyword_weight=args.keyword_weight,
    )
    
    print(f"Running Eval: Method={cfg.search_method}, Rerank={cfg.reranking_enable}, Mode={cfg.reranking_mode}...")
    if args.debug:
        print("DEBUG MODE ON")
    
    metrics = evaluate_csv(cfg=cfg, csv_path=args.csv, out_result_path=args.out, debug=args.debug)
    print(
        f"Result: count={int(metrics['count'])}, hit@{args.top_k}={metrics['hit@K']:.4f}, top1={metrics['top1']:.4f}"
    )


if __name__ == "__main__":
    main()
