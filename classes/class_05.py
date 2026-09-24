# %%
import pandas as pd
import numpy as np
import pickle

from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# %%
url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# %%
df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
df["seniorcitizen"] = df["seniorcitizen"].astype(str)

# %%
for c in df.select_dtypes(include=["object"]).columns:
    df[c] = df[c].str.lower().str.strip().str.replace(" ", "_")

df["churn"] = df["churn"].replace({"yes": 1, "no": 0})
df = df.fillna(0)

target = "churn"
features = df.columns.drop(target)

categorical_columns = list(df[features].select_dtypes(include=["object"]).columns)
numerical_columns = list(df[features].select_dtypes(exclude=["object"]).columns)

# %%
df_full_train, df_test = model_selection.train_test_split(
    df, 
    test_size=0.2, 
    random_state=42,
    stratify=df["churn"]
    )

df_train, df_val = model_selection.train_test_split(
    df_full_train, 
    test_size=0.25, 
    random_state=42,
    stratify=df_full_train["churn"]
    )

# %%
y_full_train = df_full_train[target].copy()
y_train = df_train[target].copy()
y_val = df_val[target].copy()
y_test = df_test[target].copy()

# %%
def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical_columns + numerical_columns].to_dict(orient="records")
 
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[categorical_columns + numerical_columns].to_dict(orient="records")

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

# %%
dv, model = train(df_full_train, y_full_train, C = 10)
y_pred = predict(df_test, dv, model)

# %%
auc = roc_auc_score(y_test, y_pred)
auc

# %%
output_file = "model_C=10.bin"

with open(f"../files/{output_file}", "wb") as f_out:
    pickle.dump((dv, model), f_out)

# %%
model_file = "model_C=10.bin"

with open(f"../files/{model_file}", "rb") as f_in:
    dv, model = pickle.load(f_in)

# %%
customer = (df.iloc[np.random.randint(len(df))]).drop(["customerid", "churn"]).to_dict()
customer

# %%
X = dv.transform([customer])
model.predict_proba(X)[0, 1]