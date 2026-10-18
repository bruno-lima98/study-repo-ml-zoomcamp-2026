"""Questão 1 — inspeciona as entradas e saídas do grafo ONNX.

Uso:
    python inspect_model.py
"""
import onnxruntime as ort

MODEL_PATH = "hair_classifier_v1.onnx"  # o .onnx.data precisa estar na mesma pasta


def main():
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

    print("Entradas:")
    for inp in session.get_inputs():
        print(f"  nome={inp.name!r}  shape={inp.shape}  tipo={inp.type}")

    print("Saídas:")
    for out in session.get_outputs():
        print(f"  nome={out.name!r}  shape={out.shape}  tipo={out.type}")


if __name__ == "__main__":
    main()
