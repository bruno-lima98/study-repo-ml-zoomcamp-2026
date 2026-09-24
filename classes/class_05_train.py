# LIBRARIES
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


# PARAMETERS
C = 10
n_splits = 5
output_file = f"model_C={C}.bin"


# DATA PREPARATION
url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(url)

df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
df["seniorcitizen"] = df["seniorcitizen"].astype(str)

for c in df.select_dtypes(include=["object"]).columns:
    df[c] = df[c].str.lower().str.strip().str.replace(" ", "_")

df["churn"] = df["churn"].replace({"yes": 1, "no": 0})
df = df.fillna(0)

target = "churn"
features = df.columns.drop(target)

categorical_columns = list(df[features].select_dtypes(include=["object"]).columns)
numerical_columns = list(df[features].select_dtypes(exclude=["object"]).columns)

df_full_train, df_test = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42,
    stratify=df["churn"]
    )

df_train, df_val = train_test_split(
    df_full_train, 
    test_size=0.25, 
    random_state=42,
    stratify=df_full_train["churn"]
    )

y_full_train = df_full_train[target].copy()
y_train = df_train[target].copy()
y_val = df_val[target].copy()
y_test = df_test[target].copy()

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


# MODEL VALIDATION
print(f"Doing validation with C={C}")

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f"AUC on fold {fold} is {round(auc,4)}")
    fold = fold + 1


print("Validation results:")
print(f"C={C} | {round(np.mean(scores),3)} +- {round(np.std(scores),3)}")

# MODEL TRAINING
print('Training the final model...')

dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)

print(f'AUC = {round(auc,4)}')

# MODEL SAVING
with open(f"../files/{output_file}", "wb") as f_out:
    pickle.dump((dv, model), f_out)

print(f'The model is saved in {output_file}')