# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

# %%
url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-06-trees/CreditScoring.csv"

df = pd.read_csv(url)
df.head()

# %%
df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

# %%
status_column = {
    1: "ok",
    2: "default",
    0: "unknown"
}

home_column = {
    1: "rent",
    2: "owner",
    3: "private",
    4: "ignore",
    5: "parents",
    6: "other",
    0: "unknown"
}

marital_column = {
    1: "single",
    2: "married",
    3: "widow",
    4: "separeted",
    5: "divorced",
    0: "unknown"
}

records_column = {
    1: "no",
    2: "yes",
    0: "unknown"
}

job_column = {
    1: "fixed",
    2: "partime",
    3: "freelance",
    4: "others",
    0: "unknown"
}

# %%
categorical_columns = ["status", "home", "marital", "records", "job"]

maping = {
    "status": status_column, 
    "home": home_column, 
    "marital": marital_column, 
    "records": records_column, 
    "job": job_column
}

for col in categorical_columns:
    df[col] = df[col].map(maping[col])

df.head()

# %%
df.describe().round().T

# %%
replacing = ["income", "assets", "debt"]

for col in replacing:
    df[col] = df[col].replace(to_replace=99999999, value=np.nan)

df.describe().round().T
 
# %%
df["status"].value_counts()

# %%
df = df[df["status"] != "unknown"].reset_index(drop=True)
df["status"].value_counts()

# %%
status_column = {
    "ok": 0,
    "default": 1
}

df["status"] = df["status"].map(status_column)
df["status"].value_counts()

# %%
df_full_train, df_test = train_test_split(
                                        df, 
                                        test_size=0.2, 
                                        random_state=42, 
                                        stratify=df["status"]
                                        )

df_train, df_val = train_test_split(
                                df_full_train, 
                                test_size=0.25, 
                                random_state=42, 
                                stratify=df_full_train["status"]
                                )

(len(df_full_train), len(df_train), len(df_val), len(df_test))

# %%
target = "status"
features = df.columns.drop(target)

# %%
y_train = df_train[target].reset_index(drop=True).copy()
y_val = df_val[target].reset_index(drop=True).copy()
y_test = df_test[target].reset_index(drop=True).copy()

df_train = df_train[features].reset_index(drop=True).copy()
df_val = df_val[features].reset_index(drop=True).copy()
df_test = df_test[features].reset_index(drop=True).copy()

# %%
train_dicts = df_train.fillna(0).to_dict(orient='records')

# %%
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

# %%
dt = DecisionTreeClassifier(max_depth=4)
dt.fit(X_train, y_train)

# %%
val_dicts = df_val.fillna(0).to_dict(orient='records')
X_val = dv.transform(val_dicts)

# %%
y_pred = dt.predict_proba(X_train)[:, 1]
auc = roc_auc_score(y_train, y_pred)
print('train:', auc)

y_pred = dt.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
print('val:', auc)

# %%
