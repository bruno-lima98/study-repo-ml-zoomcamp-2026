# %%
import pandas as pd
import numpy as np
from IPython.display import display

from sklearn import model_selection

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