# %%
import pandas as pd
import numpy as np
from IPython.display import display
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# %%
url = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv"

df = pd.read_csv(url)

df.head()

# %%
df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
df.head().T

# %%
df.dtypes

# %%
target = df.columns[-1]
features = df.columns [:-1]

categorical_columns = list(df[features].dtypes[df[features].dtypes == "object"].index)
numerical_columns = list(df[features].dtypes[df[features].dtypes != "object"].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.strip().str.replace(" ", "_")

# %%
df[target] = df[target].replace({"yes": 1, "no": 0})
df.head().T

# %%
df.shape

# %%
df.isnull().sum()

# %%
for c in categorical_columns:
    df[c] = df[c].fillna("NA")

for c in numerical_columns:
    df[c] = df[c].fillna(0)

# %%
# Q1:
df["industry"].value_counts()

# %%
# Q2:
corr = df[numerical_columns].corr().round(3)
corr

# %%
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

# %%
y_full_train = df_full_train["converted"].copy()
y_train = df_train["converted"].copy()
y_val = df_val["converted"].copy()
y_test = df_test["converted"].copy()

del df_full_train["converted"]
del df_train["converted"]
del df_val["converted"]
del df_test["converted"]

# %%
# Q3:
def mutual_info_convert_score(series):
    return mutual_info_score(series, y_full_train)

mutual = df_full_train[categorical_columns].apply(mutual_info_convert_score)
mutual.sort_values(ascending=False).round(2)

# %%
# Q4:
train_dicts = df_train[categorical_columns + numerical_columns].to_dict(orient="records")
val_dicts = df_val[categorical_columns + numerical_columns].to_dict(orient="records")

dv = DictVectorizer(sparse=False)

# %%
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

# %%
list(dv.get_feature_names_out())

# %%
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

print(f"w0 = {model.intercept_[0].round(4)}")
print(f"w = {model.coef_[0].round(3)}")

# %%
y_pred = model.predict_proba(X_val)[:, 1]
churn_decision = (y_pred >= 0.5)

(y_val == churn_decision).mean()

# %%
base_acc = (y_val == churn_decision).mean()

# %%
# Q5:
testing_features = ["lead_source", "number_of_courses_viewed", "interaction_count"]

for feature in testing_features:
    df_train_small = df_train.copy()
    df_val_small = df_val.copy()

    del df_train_small[feature]
    del df_val_small[feature]

    features = df_train_small.columns [:-1]

    categorical_columns_small = list(df_train_small[features].dtypes[df_train_small[features].dtypes == "object"].index)
    numerical_columns_small = list(df_train_small[features].dtypes[df_train_small[features].dtypes != "object"].index)

    train_small_dicts = df_train_small[categorical_columns_small + numerical_columns_small].to_dict(orient="records")
    val_small_dicts = df_val_small[categorical_columns_small + numerical_columns_small].to_dict(orient="records")

    dv = DictVectorizer(sparse=False)

    X_train_small = dv.fit_transform(train_small_dicts)
    X_val_small = dv.transform(val_small_dicts)

    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train_small, y_train)

    y_pred = model.predict_proba(X_val_small)[:, 1]
    churn_decision = (y_pred >= 0.5)

    acc = (y_val == churn_decision).mean()

    print(f"Feature removida: {feature}")
    print(f"Accuracy: {acc.round(4)} | Accuracy base: {base_acc.round(4)} | Diferença: {round(base_acc - acc, 4)}")
    print()

# %%
# Q6:
C = [0.000001, 0.00001, 0.0001, 0.001] 

for c in C:
    model = LogisticRegression(solver='liblinear', C=c, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_val)[:, 1]
    churn_decision = (y_pred >= 0.5)

    acc = (y_val == churn_decision).mean()

    print(f"C = {c}")
    print(f"Accuracy: {acc.round(4)}")
    print()

# %%
