# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor 

import xgboost as xgb

# %%
url = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"

df = pd.read_csv(url)
df.head()

# %%
df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
df["num_doors"] = df["num_doors"].astype(str)

target = "fuel_efficiency_mpg"
features = df.columns.drop(target)

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
df_full_train, df_test = train_test_split(
    df, test_size=0.2, 
    random_state=1
    )
df_train, df_val = train_test_split(
    df_full_train, 
    test_size=0.25, 
    random_state=1
    )

y_train = df_train[target].reset_index(drop=True).copy()
y_val = df_val[target].reset_index(drop=True).copy()
y_test = df_test[target].reset_index(drop=True).copy()
y_full_train = df_full_train[target].reset_index(drop=True).copy()

df_train = df_train[features].reset_index(drop=True).copy()
df_val = df_val[features].reset_index(drop=True).copy()
df_test = df_test[features].reset_index(drop=True).copy()
df_full_train = df_full_train[features].reset_index(drop=True).copy()

# %%
train_dicts = df_train.to_dict(orient="records")
val_dicts = df_val.to_dict(orient="records")

dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

# %%
# Q1:
dt = DecisionTreeRegressor(max_depth=1)
dt.fit(X_train, y_train)

print(export_text(dt, feature_names=dv.get_feature_names_out()))

# %%
# Q2:
rf = RandomForestRegressor(
    n_estimators=10,
    random_state=1,
    n_jobs=-1
    )

rf.fit(X_train, y_train)

# %%
y_pred = rf.predict(X_val)

rmse = mean_squared_error(y_val, y_pred) ** 0.5
mae = mean_absolute_error(y_val, y_pred)

print("RMSE:", round(rmse,4))
print("MAE:", round(mae,4))

# %%
# Q3:
n_estimators = [10, 50, 100, 150]

for n in n_estimators:
    rf = RandomForestRegressor(
        n_estimators=n,
        random_state=1,
        n_jobs=-1
        )

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_val)

    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    mae = mean_absolute_error(y_val, y_pred)

    print(f"n_estimators = {n} | RMSE = {rmse:.4f} | MAE = {mae:.3f}")

# %%
# Q4:
n_estimators = [10, 50, 100, 150]
max_depth = [10, 15, 20, 25]

for m in max_depth:
    score = []
    
    for n in n_estimators:    
        rf = RandomForestRegressor(
            n_estimators=n,
            random_state=1,
            n_jobs=-1,
            max_depth=m
            )

        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred) ** 0.5
        score.append(rmse)

        print(f"max_depth = {m} | n_estimators = {n} | RMSE = {rmse:.4f}")

    print(f"max_depth = {m} | RMSE mean = {statistics.mean(score):.4f}")

# %%
# Q5:
rf = RandomForestRegressor(
    n_estimators=10,
    random_state=1,
    max_depth=20,
    n_jobs=-1
    )

rf.fit(X_train, y_train)

# %%
y_pred = rf.predict(X_val)

rmse = mean_squared_error(y_val, y_pred) ** 0.5
mae = mean_absolute_error(y_val, y_pred)

print("RMSE:", round(rmse,4))
# %%
feature_importance = pd.DataFrame({
    "feature": dv.get_feature_names_out(),
    "importance": rf.feature_importances_
})

feature_importance.sort_values("importance", ascending=False)

# %%
# Q6:
xgb_features = dv.get_feature_names_out()

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(xgb_features))
dval = xgb.DMatrix(X_val, label=y_val, feature_names=list(xgb_features))

# %%
eta_scores = {}
etas = [0.1, 0.3]

watchlist = [(dtrain, "train"), (dval, "validation")]

num_boost_round = 100

for eta in etas:
    evals_result = {}

    xgb_params = {
        "eta": eta,
        "max_depth": 6,
        "min_child_weight": 1,
        
        "objective": "reg:squarederror",
        "nthread": 8,
        
        "seed": 1,
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
        "train_rmse": evals_result["train"]["rmse"],
        "val_rmse": evals_result["validation"]["rmse"]
    })

    key = f"eta_{xgb_params["eta"]}"
    eta_scores[key] = df_scores

# %%
for key, df_score in eta_scores.items():
    plt.plot(df_score["round"], df_score["val_rmse"], label=key)

plt.xlabel("Boosting Round")
plt.ylabel("rmse")
plt.legend()
plt.show()

