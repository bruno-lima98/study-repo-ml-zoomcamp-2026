"""Handler compatível com AWS Lambda.

Aceita dois formatos de evento:
  1. Direto:               {"image_url": "https://..."}
  2. Estilo API Gateway:   {"body": "{\"image_url\": \"https://...\"}"}

Resposta:
  {"statusCode": 200,
   "headers": {"Content-Type": "application/json"},
   "body": "{\"straight_probability\": 0.728, \"straight\": true}"}

A sessão do ONNX Runtime é criada fora do handler, quando o módulo é carregado.
O Lambda reaproveita o container entre invocações, então o modelo é carregado uma vez só.
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from preprocess import download_image, prepare_image

# Caminho relativo a este arquivo: funciona tanto local quanto em /var/task no Lambda
MODEL_PATH = Path(__file__).resolve().parent / "hair_classifier_v1.onnx"

session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name    # "input"
OUTPUT_NAME = session.get_outputs()[0].name  # "output"


def predict_image(image):
    x = prepare_image(image)
    result = session.run([OUTPUT_NAME], {INPUT_NAME: x})[0]
    return float(np.asarray(result).reshape(-1)[0])


def predict(image_url):
    return predict_image(download_image(image_url))


def parse_event(event):
    """Extrai o payload do evento, seja ele direto ou vindo do API Gateway."""
    if "body" in event:
        body = event["body"]
        if isinstance(body, str):
            body = json.loads(body or "{}")
        return body or {}
    return event


def response(status_code, payload):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }


def lambda_handler(event, context):
    payload = parse_event(event)
    image_url = payload.get("image_url")
    if not image_url:
        return response(400, {"error": "missing 'image_url'"})

    probability = predict(image_url)
    return response(200, {
        "straight_probability": probability,
        "straight": probability >= 0.5,
    })
