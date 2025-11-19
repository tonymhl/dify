# 使用方法
```bash
usage: dify_eval.py [-h] [--endpoint ENDPOINT] [--token TOKEN] [--csv CSV] [--top-k TOP_K] [--out OUT]

Dify hit-testing 测试与评测（Hit@K，默认K=5）

options:
  -h, --help           show this help message and exit
  --endpoint ENDPOINT  Dify hit-testing 接口完整URL，例如：https://<host>/console/api/datasets/<dataset_id>/hit-testing      
  --token TOKEN        Dify Bearer Token
  --csv CSV            测试集合CSV（列名：问题,文档名）
  --top-k TOP_K        评测时请求的TopK
  --out OUT            逐条结果（JSONL）输出路径
```

# 使用样例
```bash
python3 dify_eval.py --endpoint  https://dify-aios.gds-services.com/console/api/datasets/820d67aa-6ed6-45e0-95fa-e59285e1397c/hit-testing --token "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNmI2NjkyOTktZmVlYi00ZTM1LWJmZTMtOWVkNzY4ZjlmNTRlIiwiZXhwIjoxNzYzNTQ4OTE4LCJpc3MiOiJTRUxGX0hPU1RFRCIsInN1YiI6IkNvbnNvbGUgQVBJIFBhc3Nwb3J0In0.nO0S24vr5C6f3CRu_eCPurN1x3YRd0Q7vyzyLehbiDE" --csv test_case/test_case.csv --output output/dify_eval_results.json
```