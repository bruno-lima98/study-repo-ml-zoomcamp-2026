# %%
import pandas as pd
import numpy as np
from IPython.display import display
import random
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mutual_info_score, accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# %%
url = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv"

df = pd.read_csv(url)

df.head()

# %%
df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

target = df.columns[-1]
features = df.columns [:-1]

categorical_columns = list(df[features].dtypes[df[features].dtypes == "object"].index)
numerical_columns = list(df[features].dtypes[df[features].dtypes != "object"].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.strip().str.replace(" ", "_")

# %%
for c in categorical_columns:
    df[c] = df[c].fillna("NA")

for c in numerical_columns:
    df[c] = df[c].fillna(0)
# %%
df.isnull().sum()

# %%
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=1
)

# %%
# Q1:
scores = []

for col in numerical_columns:
    score = roc_auc_score(df_train[target], df_train[col])
    scores.append(score)

    print(col, round(score,4))

# %%
# Q2:
X_full_train = df_full_train[features].copy()
y_full_train = df_full_train[target].copy()

X_train = df_train[features].copy()
y_train = df_train[target].copy()

X_val = df_val[features].copy()
y_val = df_val[target].copy()

X_test = df_test[features].copy()
y_test = df_test[target].copy()

# %%
train_dicts = X_train[categorical_columns + numerical_columns].to_dict(orient="records")
val_dicts = X_val[categorical_columns + numerical_columns].to_dict(orient="records")

dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_val)[:, 1]
y_pred_final = model.predict(X_val)[:]
churn_decision = (y_pred >= 0.5)

# %%
roc_auc_score(y_val, y_pred).round(3)

# %%
# Q3
threshold = np.linspace(0, 1, 101)
precisions =[]
recalls = []

for t in threshold:
    churn_decision = (y_pred >= t)

    cm = confusion_matrix(y_val, churn_decision)

    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    precisions.append(precision)
    recalls.append(recall)

# %%
plt.plot(threshold, precisions, label="Precision")
plt.plot(threshold, recalls, label="Recall")

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()

plt.show()

# %%
n = len(precisions)
diff = np.ones(n)

for i in range(n):
    diff[i] = abs(precisions[i] - recalls[i])

v_diff = np.nanmin(diff)
idx = np.nanargmin(diff)

print(idx, round(v_diff,4), threshold[idx])

# %%
# Q4
f1 = np.ones(n)

for i in range(n):
    f1[i] = (2 * precisions[i] * recalls[i])/(precisions[i] + recalls[i])

max_f1 = np.nanmax(f1)
idx = np.nanargmax(f1)

print(threshold[idx], max_f1)

plt.plot(threshold, f1, label="F1 Score")

plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.legend()

plt.show()

# %%
# Q5
def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical_columns + numerical_columns].to_dict(orient="records")
 
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[categorical_columns + numerical_columns].to_dict(orient="records")

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

# %%
n_splits = 5

scores = []

kfold = KFold(n_splits=5, shuffle=True, random_state=1)

for train_idx, val_idx in tqdm(kfold.split(df_full_train), total=n_splits):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train[target].values
    y_val = df_val[target].values

    dv, model = train(df_train, y_train)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(round(auc, 4))

np.std(scores).round(3)

# %%
# Q6
C = [0.000001, 0.001, 1]

for c in C:
    scores = []

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in tqdm(kfold.split(df_full_train), total=n_splits):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train[target].values
        y_val = df_val[target].values

        dv, model = train(df_train, y_train, C=c)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print(f"C = {c} | {np.mean(scores).round(4)} +- {np.std(scores).round(4)}")
