# %%
import pandas as pd
import numpy as np
from IPython.display import display
import random

from sklearn import model_selection
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# %%
url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(url)

df.head()

# %%
df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
df.head().T

# %%
df.dtypes

# %%
df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
df["seniorcitizen"] = df["seniorcitizen"].astype(str)

# %%
categorical_columns = list(df.dtypes[df.dtypes == "object"].index)
numerical_columns = list(df.dtypes[df.dtypes != "object"].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.strip().str.replace(" ", "_")

# %%
df["churn"] = df["churn"].replace({"yes": 1, "no": 0})
df.head().T

# %%
df["churn"].value_counts()
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
print(len(df))
print(len(df_full_train))
print(len(df_train))
print(len(df_val))
print(len(df_test))

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
df_full_train.isnull().sum()

# %%
churn_counts = df["churn"].value_counts()

resultado = pd.DataFrame({
    "total": churn_counts,
    "percentual": df["churn"].value_counts(normalize=True)
})
resultado

# %%
categorical_columns.remove(target)
categorical_columns.remove("customerid")

# %%
for col in categorical_columns:
    print(col)
    print(df_full_train[col].nunique())
    print(df_full_train[col].unique()[0:5])
    print()

# %%
global_churn = df_full_train["churn"].mean()

for col in categorical_columns:
    print(col)
    df_group = df_full_train.groupby(col)["churn"].agg(["count","mean"])
    df_group["diff"] = df_group["mean"] - global_churn
    df_group["risk"] = df_group["mean"] / global_churn
    display(df_group)
    print()
# %%
def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train["churn"])

mutual = df_full_train[categorical_columns].apply(mutual_info_churn_score)
mutual.sort_values(ascending=False)

# %%
df_full_train[numerical_columns].corrwith(df_full_train["churn"])

# %%
train_dicts = X_train[categorical_columns + numerical_columns].to_dict(orient="records")
val_dicts = X_val[categorical_columns + numerical_columns].to_dict(orient="records")

dv = DictVectorizer(sparse=False)

# %%
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

# %%
list(dv.get_feature_names_out())

# %%
model = LogisticRegression()
model.fit(X_train, y_train)

# %%
model.intercept_[0].round(4) # >> coeficiente independente

# %%
model.coef_[0].round(3) # >> coeficientes das features

# %%
y_pred = model.predict_proba(X_val)[:, 1]
churn_decision = (y_pred >= 0.5)

(y_val == churn_decision).mean()

# %%
dict(zip(dv.get_feature_names_out(), model.coef_[0].round(3)))

# %%
full_train_dicts = X_full_train[categorical_columns + numerical_columns].to_dict(orient="records")
dv = DictVectorizer(sparse=False)

X_full_train = dv.fit_transform(full_train_dicts)

# %% 
model = LogisticRegression()
model.fit(X_full_train, y_full_train)

# %%
test_dicts = X_test[categorical_columns + numerical_columns].to_dict(orient="records")

X_test = dv.transform(test_dicts)

# %%
y_pred = model.predict_proba(X_test)[:, 1]
churn_decision = (y_pred >= 0.5)

(churn_decision == y_test).mean()

# %%
rand = random.randint(0, len(X_test))
customer = test_dicts[rand]

X_small = dv.transform([customer])

print(f"Probabilidade de churn (modelo): {model.predict_proba(X_small)[0, 1].round(4)}")
print(f"Decisão (modelo): {model.predict(X_small)[0]}")
print(f"Valor real: {y_test.iloc[rand]}")