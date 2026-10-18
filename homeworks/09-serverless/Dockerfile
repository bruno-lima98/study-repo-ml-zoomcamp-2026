FROM public.ecr.aws/lambda/python:3.13

# Dependências primeiro: assim o Docker reaproveita essa camada
# quando só o código muda
COPY requirements-lambda.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt

# O par ONNX precisa ficar junto, com esses nomes exatos
COPY hair_classifier_v1.onnx hair_classifier_v1.onnx.data ${LAMBDA_TASK_ROOT}/

COPY preprocess.py lambda_function.py ${LAMBDA_TASK_ROOT}/

# arquivo.função que o Lambda chama
CMD ["lambda_function.lambda_handler"]
