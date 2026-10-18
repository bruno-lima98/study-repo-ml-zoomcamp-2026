"""Testa o handler localmente, sem Docker.

Uso:
    python smoke_test.py
"""
import json

from lambda_function import lambda_handler

SAMPLE_URL = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"


def check(name, event):
    result = lambda_handler(event, None)
    body = json.loads(result["body"])
    print(f"[{name}] status={result['statusCode']} body={body}")

    assert result["statusCode"] == 200
    assert 0.0 <= body["straight_probability"] <= 1.0
    assert isinstance(body["straight"], bool)
    assert body["straight"] == (body["straight_probability"] >= 0.5)
    return body


def main():
    direct = check("evento direto", {"image_url": SAMPLE_URL})
    gateway = check("API Gateway", {"body": json.dumps({"image_url": SAMPLE_URL})})

    # Os dois formatos têm que dar o mesmo resultado
    assert direct == gateway

    print("OK")


if __name__ == "__main__":
    main()
