import requests

url = "http://localhost:9696/predict"

customer_id = "xyz-123"
customer = {
    "gender": "male",
    "seniorcitizen": "0",
    "partner": "no",
    "dependents": "no",
    "tenure": 13,
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "yes",
    "onlinebackup": "no",
    "deviceprotection": "no",
    "techsupport": "yes",
    "streamingtv": "yes",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "credit_card_(automatic)",
    "monthlycharges": 45.55,
    "totalcharges": 597.0
    }

response = requests.post(url, json=customer).json()
print(response)

if response["Churn"] == True:
    print(f"Enviando email promocional ao cliente {customer_id}.")
else:
    print(f"Não é necessário email promocional ao cliente {customer_id}.")

# waitress-serve --listen=0.0.0.0:9696 class_05_predict:app
