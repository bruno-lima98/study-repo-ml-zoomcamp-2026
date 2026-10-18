"""Pré-processamento da imagem — exatamente o do enunciado.

Pontos que fazem parte da "interface" do modelo (não mude):
- converter para RGB
- resize para 200x200 com BILINEAR (explícito!)
- float32, dividir por 255
- normalizar com média/desvio do ImageNet
- trocar de HWC para CHW e adicionar a dimensão de batch -> (1, 3, 200, 200)
"""
from io import BytesIO
from urllib import request

import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def download_image(url):
    with request.urlopen(url, timeout=30) as response:
        return Image.open(BytesIO(response.read())).convert("RGB")


def load_local_image(path):
    """Útil para testar sem internet: abre um arquivo local do mesmo jeito."""
    with open(path, "rb") as f:
        return Image.open(BytesIO(f.read())).convert("RGB")


def prepare_image(image):
    image = image.resize((200, 200), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    return np.transpose(array, (2, 0, 1))[None, ...]
