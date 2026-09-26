# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

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
y_full_train = df_full_train[target].reset_index(drop=True).copy()

df_train = df_train[features].reset_index(drop=True).copy()
df_val = df_val[features].reset_index(drop=True).copy()
df_test = df_test[features].reset_index(drop=True).copy()
df_full_train = df_full_train[features].reset_index(drop=True).copy()

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
max_depth = [1, 2, 3, 4, 5, 10, 20]

for d in max_depth:
    dt = DecisionTreeClassifier(max_depth=d)
    dt.fit(X_train, y_train)

    y_pred = dt.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    print(f'Max Depth = {d:^4} | AUC:', round(auc,4))

# %%
max_depth = [1, 2, 3, 4, 5, 10, 20]
min_sample = [1, 2, 3, 4, 5, 10, 14, 20, 50, 100, 200, 500]

scores = []

for d in max_depth:
    for m in min_sample:
        dt = DecisionTreeClassifier(max_depth=d, min_samples_leaf=m)
        dt.fit(X_train, y_train)

        y_pred = dt.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((d, m, auc))

best_d, best_m, auc_max = max(scores, key=lambda x: x[2])

print(f"Best max_depth: {best_d}")
print(f"Best min_samples_leaf: {best_m}")
print(f"Best AUC: {auc_max:.4f}")

# %%
dt = DecisionTreeClassifier(max_depth=best_d, min_samples_leaf=best_m)
dt.fit(X_train, y_train)

# %%
rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X_train, y_train)

# %%
y_pred = rf.predict_proba(X_val)[:, 1]

# %%
roc_auc_score(y_val, y_pred)

# %%
n_estimators = np.arange(10, 201, 1)
scores = []

for n in n_estimators:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)

    scores.append((n, auc))
    print(f"n = {n:^3} | AUC = {auc:.4f}")
    
best_n, auc_max = max(scores, key=lambda x: x[1])

print(f"Best n: {best_n}")
print(f"Best AUC: {auc_max:.4f}")
    
# %%
df_scores = pd.DataFrame(scores, columns=["n", "auc"])

plt.plot(df_scores["n"], df_scores["auc"])

# %%
n_estimators = np.arange(10, 201, 10)
max_depth = [5, 10, 15]

scores = []

for m in max_depth:
    for n in n_estimators:
        rf = RandomForestClassifier(n_estimators=n, max_depth=m, random_state=42)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((n, m, auc))
        print(f"n = {n:^3} | max_depth={m:^2} | AUC = {auc:.4f}")
    
best_n, best_m, auc_max = max(scores, key=lambda x: x[2])

print(f"Best n: {best_n}")
print(f"Best max_depth: {best_m}")
print(f"Best AUC: {auc_max:.4f}")

# %%
df_scores = pd.DataFrame(scores, columns=["n", "m", "auc"])

for m in max_depth:
    df_subset = df_scores[df_scores["m"] == m]
    plt.plot(df_subset["n"], df_subset["auc"], label=f"max_depth = {m}")

plt.legend()

# %%
min_sample = [1, 3, 5, 10, 50, 100]
scores = []

for min in min_sample:
    for n in n_estimators:
        rf = RandomForestClassifier(
            n_estimators=n, 
            max_depth=best_m, 
            random_state=42,
            min_samples_leaf=min
            )
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((n, min, auc))
        print(f"n = {n:^3} | min_samples_leaf={min:^2} | AUC = {auc:.4f}")
    
best_n, best_min, auc_max = max(scores, key=lambda x: x[2])

print(f"Best n: {best_n}")
print(f"Best max_depth: {best_m}")
print(f"Best min_samples_leaf: {best_min}")
print(f"Best AUC: {auc_max:.4f}")

# %%
df_scores = pd.DataFrame(scores, columns=["n", "min", "auc"])

for min in min_sample:
    df_subset = df_scores[df_scores["min"] == min]
    plt.plot(df_subset["n"], df_subset["auc"], label=f"min_samples_leaf = {min}")

plt.legend()

# %%
rf = RandomForestClassifier(
            n_estimators=best_n, 
            max_depth=best_m, 
            random_state=42,
            min_samples_leaf=best_min,
            n_jobs=-1
            )

rf.fit(X_train, y_train)

# %%
features = dv.get_feature_names_out()

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(features))
dval = xgb.DMatrix(X_val, label=y_val, feature_names=list(features))

# %%
xgb_params = {
    "eta": 0.3,
    "max_depth": 6,
    "min_child_weight": 1,
    
    "objective": "binary:logistic",
    "nthread": 8,
    
    "seed": 42,
    "verbosity": 1
}

model = xgb.train(xgb_params, dtrain, num_boost_round=200)

# %%
y_pred = model.predict(dval)
roc_auc_score(y_val, y_pred)

# %%
watchlist = [(dtrain, "train"), (dval, "validation")]
evals_result = {}
num_boost_round = 200

xgb_params = {
    "eta": 0.3,
    "max_depth": 6,
    "min_child_weight": 1,
    
    "objective": "binary:logistic",
    "nthread": 8,
    "eval_metric": "auc",
    
    "seed": 42,
    "verbosity": 1
}

model = xgb.train(
    xgb_params, 
    dtrain, 
    num_boost_round=num_boost_round,
    evals=watchlist,
    verbose_eval=5,
    evals_result=evals_result
    )

# %%
df_scores = pd.DataFrame({
    "round": range(1, num_boost_round+1),
    "train_auc": evals_result["train"]["auc"],
    "val_auc": evals_result["validation"]["auc"]
})

# plt.plot(df_scores["round"], df_scores["train_auc"], label="train")
plt.plot(df_scores["round"], df_scores["val_auc"], label="validation")

plt.xlabel("Boosting Round")
plt.ylabel("AUC")
plt.legend()
plt.show()

# %%
eta_scores = {}
etas = [0.01, 0.05, 0.1, 0.3, 0.5, 1]

watchlist = [(dtrain, "train"), (dval, "validation")]

num_boost_round = 200

for eta in etas:
    evals_result = {}

    xgb_params = {
        "eta": eta,
        "max_depth": 6,
        "min_child_weight": 1,
        
        "objective": "binary:logistic",
        "nthread": 8,
        "eval_metric": "auc",
        
        "seed": 42,
        "verbosity": 1
    }

    model = xgb.train(
        xgb_params, 
        dtrain, 
        num_boost_round=num_boost_round,
        evals=watchlist,
        verbose_eval=5,
        evals_result=evals_result
        )

    df_scores = pd.DataFrame({
        "round": range(1, num_boost_round+1),
        "train_auc": evals_result["train"]["auc"],
        "val_auc": evals_result["validation"]["auc"]
    })

    key = f"eta_{xgb_params["eta"]}"
    eta_scores[key] = df_scores

# %%
for key, df_score in eta_scores.items():
    plt.plot(df_score["round"], df_score["val_auc"], label=key)

plt.xlabel("Boosting Round")
plt.ylabel("AUC")
plt.legend()
plt.show()

# %%
etas = ["eta_0.01", "eta_0.05", "eta_0.3", "eta_1"]

for eta in etas:
    df_score = eta_scores[eta]
    plt.plot(df_score["round"], df_score["val_auc"], label=eta)

plt.xlabel("Boosting Round")
plt.ylabel("AUC")
plt.legend(loc="lower right")
plt.show()

# %%
best_eta = 0.05
num_boost_round = 200

max_depth_scores = {}
max_depths = [1, 2, 3, 5, 10]

watchlist = [(dtrain, "train"), (dval, "validation")]

for max_d in max_depths:
    evals_result = {}

    xgb_params = {
        "eta": best_eta,
        "max_depth": max_d,
        "min_child_weight": 1,
        
        "objective": "binary:logistic",
        "nthread": 8,
        "eval_metric": "auc",
        
        "seed": 42,
        "verbosity": 1
    }

    model = xgb.train(
        xgb_params, 
        dtrain, 
        num_boost_round=num_boost_round,
        evals=watchlist,
        verbose_eval=5,
        evals_result=evals_result
        )

    df_scores = pd.DataFrame({
        "round": range(1, num_boost_round+1),
        "train_auc": evals_result["train"]["auc"],
        "val_auc": evals_result["validation"]["auc"]
    })

    key = f"max_depth_{xgb_params["max_depth"]}"
    max_depth_scores[key] = df_scores

# %%
for key, df_score in max_depth_scores.items():
    plt.plot(df_score["round"], df_score["val_auc"], label=key)

plt.xlabel("Boosting Round")
plt.ylabel("AUC")
plt.ylim(bottom=0.78)
plt.legend()
plt.show()

# %%
best_eta = 0.05
best_max_depth = 3
num_boost_round = 200

min_childs_scores = {}
min_childs = [1, 5, 10, 20, 50, 100]

watchlist = [(dtrain, "train"), (dval, "validation")]

for min_c in min_childs:
    evals_result = {}

    xgb_params = {
        "eta": best_eta,
        "max_depth": best_max_depth,
        "min_child_weight": min_c,
        
        "objective": "binary:logistic",
        "nthread": 8,
        "eval_metric": "auc",
        
        "seed": 42,
        "verbosity": 1
    }

    model = xgb.train(
        xgb_params, 
        dtrain, 
        num_boost_round=num_boost_round,
        evals=watchlist,
        verbose_eval=5,
        evals_result=evals_result
        )

    df_scores = pd.DataFrame({
        "round": range(1, num_boost_round+1),
        "train_auc": evals_result["train"]["auc"],
        "val_auc": evals_result["validation"]["auc"]
    })

    key = f"min_child_weight_{xgb_params["min_child_weight"]}"
    min_childs_scores[key] = df_scores

# %%
for key, df_score in min_childs_scores.items():
    plt.plot(df_score["round"], df_score["val_auc"], label=key)

plt.xlabel("Boosting Round")
plt.ylabel("AUC")
plt.ylim(bottom=0.82)
plt.legend()
plt.show()

# %%
num_boost_round = 1000

best_eta = 0.05
best_max_depth = 3
best_min_child = 10

xgb_params = {
    "eta": best_eta,
    "max_depth": best_max_depth,
    "min_child_weight": best_min_child,
    
    "objective": "binary:logistic",
    "nthread": 8,
    "eval_metric": "auc",
    
    "seed": 42,
    "verbosity": 1
    }

model = xgb.train(
        xgb_params, 
        dtrain, 
        num_boost_round=num_boost_round,
        evals=watchlist,
        verbose_eval=5,
        evals_result=evals_result
        )

df_scores = pd.DataFrame({
    "round": range(1, num_boost_round+1),
    "train_auc": evals_result["train"]["auc"],
    "val_auc": evals_result["validation"]["auc"]
    })

# %%
# plt.plot(df_scores["round"], df_scores["train_auc"], label="train")
plt.plot(df_scores["round"], df_scores["val_auc"], label="validation")

plt.xlabel("Boosting Round")
plt.ylabel("AUC")
plt.legend()
plt.show()

# %%
best_d = 10
best_m = 20

dt = DecisionTreeClassifier(max_depth=best_d, min_samples_leaf=best_m)
dt.fit(X_train, y_train)

best_n = 100
best_m = 10
best_min = 10

rf = RandomForestClassifier(
            n_estimators=best_n, 
            max_depth=best_m, 
            random_state=42,
            min_samples_leaf=best_min,
            n_jobs=-1
            )

rf.fit(X_train, y_train)

num_boost_round = 1000
best_eta = 0.05
best_max_depth = 3
best_min_child = 10

xgb_params = {
    "eta": best_eta,
    "max_depth": best_max_depth,
    "min_child_weight": best_min_child,
    
    "objective": "binary:logistic",
    "nthread": 8,
    "eval_metric": "auc",
    
    "seed": 42,
    "verbosity": 1
    }

model = xgb.train(
        xgb_params, 
        dtrain, 
        num_boost_round=num_boost_round
        )

# %%
y_pred = dt.predict_proba(X_val)[:,1]
auc = roc_auc_score(y_val, y_pred)
print(f"Decision Tree | AUC = {auc:.4f}")

y_pred = rf.predict_proba(X_val)[:,1]
auc = roc_auc_score(y_val, y_pred)
print(f"Random Forest | AUC = {auc:.4f}")

y_pred = model.predict(dval)
auc = roc_auc_score(y_val, y_pred)
print(f"XGBoost | AUC = {auc:.4f}")

# %%
full_train_dicts = df_full_train.fillna(0).to_dict(orient='records')
test_dicts = df_test.fillna(0).to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(full_train_dicts)
X_test = dv.transform(test_dicts)

# %%
features = dv.get_feature_names_out()

dfulltrain = xgb.DMatrix(
    X_full_train, 
    label=y_full_train, 
    feature_names=list(features)
    )

dtest = xgb.DMatrix(
    X_test, 
    feature_names=list(features)
    )

# %%
model = xgb.train(
        xgb_params, 
        dfulltrain, 
        num_boost_round=num_boost_round
        )

y_pred = model.predict(dtest)
auc = roc_auc_score(y_test, y_pred)
print(f"XGBoost | AUC = {auc:.4f}")

df_scores = pd.DataFrame({
    "round": range(1, num_boost_round+1),
    "train_auc": evals_result["train"]["auc"],
    "val_auc": evals_result["validation"]["auc"]
    })

# %%
# plt.plot(df_scores["round"], df_scores["train_auc"], label="train")
plt.plot(df_scores["round"], df_scores["val_auc"], label=f"Validation = {auc:.4f}")

plt.xlabel("Boosting Round")
plt.ylabel("AUC")
plt.legend()
plt.show()
# %%
