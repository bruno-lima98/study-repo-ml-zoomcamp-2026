# %%
import pandas as pd
import numpy as np
from IPython.display import display
import random
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from sklearn import model_selection
from sklearn.metrics import mutual_info_score, accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# %%
url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(url)

df.head()

# %%
df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
df["seniorcitizen"] = df["seniorcitizen"].astype(str)

# %%
categorical_columns = list(df.dtypes[df.dtypes == "object"].index)
numerical_columns = list(df.dtypes[df.dtypes != "object"].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.strip().str.replace(" ", "_")

df["churn"] = df["churn"].replace({"yes": 1, "no": 0})
df = df.fillna(0)

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
target = df.columns[-1]
features = df.columns [:-1]

X_full_train = df_full_train[features].copy()
y_full_train = df_full_train[target].copy()

X_train = df_train[features].copy()
y_train = df_train[target].copy()

X_val = df_val[features].copy()
y_val = df_val[target].copy()

X_test = df_test[features].copy()
y_test = df_test[target].copy()

# %%
categorical_columns.remove(target)
categorical_columns.remove("customerid")

global_churn = df_full_train["churn"].mean()
# %%
train_dicts = X_train[categorical_columns + numerical_columns].to_dict(orient="records")
val_dicts = X_val[categorical_columns + numerical_columns].to_dict(orient="records")

dv = DictVectorizer(sparse=False)

# %%
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

# %%
model = LogisticRegression()
model.fit(X_train, y_train)

# %%
y_pred = model.predict_proba(X_val)[:, 1]
y_pred_final = model.predict(X_val)[:]
churn_decision = (y_pred >= 0.5)

(y_val == churn_decision).mean()

# %%
scores = []
thresholds = np.linspace(0, 1, 21)

for t in thresholds:
    score = accuracy_score(y_val, y_pred >= t)
    scores.append(score)

plt.plot(thresholds, scores)

# %%
confusion_matrix(y_val, y_pred_final)

# %%
precision_score(y_val, y_pred_final)

# %%
recall_score(y_val, y_pred_final)

# %%
fpr, tpr, thresholds = roc_curve(y_val, y_pred)
auc = roc_auc_score(y_val, y_pred)

plt.figure(figsize=(5,5))

plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1], [0,1], color="black", linestyle="dashed")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("images/class_04_01_roc_curve")

plt.show()

# %%
precision, recall, thresholds = precision_recall_curve(y_val, y_pred)
ap = average_precision_score(y_val, y_pred)

plt.figure(figsize=(5,5))

plt.plot(recall, precision, label=f"AP = {ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig("images/class_04_02_precision_recall_curve")

plt.show()

# %%
def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical_columns + numerical_columns].to_dict(orient="records")
 
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=100000)
    model.fit(X_train, y_train)

    return dv, model

df, model = train(df_train, y_train, C=000.1)

# %%
def predict(df, dv, model):
    dicts = df[categorical_columns + numerical_columns].to_dict(orient="records")

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

y_pred = predict(df_val, dv, model)

# %%
kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=42)

# %%
n_splits = 10
C = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10]

for c in C:
    scores = []

    kfold = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in tqdm(kfold.split(df_full_train), total=n_splits):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train["churn"].values
        y_val = df_val["churn"].values

        dv, model = train(df_train, y_train, c)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print(f"C = {c} | {np.mean(scores).round(4)} +- {np.std(scores).round(4)}")

# %%
dv, model = train(df_full_train, y_full_train, C = 10)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
auc

# %%
