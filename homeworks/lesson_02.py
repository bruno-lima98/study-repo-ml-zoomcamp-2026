# %%

import pandas as pd
import numpy as np

# %%
url = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"

df = pd.read_csv(url)

df.head()

# %%
columns = [
    "engine_displacement",
    "horsepower",
    "vehicle_weight",
    "model_year",
    "fuel_efficiency_mpg"
]

df = df[columns]

# %%
# Q1
df.isnull().sum()

# %%
# Q2
df["horsepower"].median()

# %%
n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

np.random.seed(42)
idx = np.arange(n)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val:]]

# %%
features = [
    "engine_displacement",
    "horsepower",
    "vehicle_weight",
    "model_year"
]

target = "fuel_efficiency_mpg"

# %%
X_train = df_train[features].copy()
X_val = df_val[features].copy()
X_test = df_test[features].copy()

y_train = df_train[target].copy()
y_val = df_val[target].copy()
y_test = df_test[target].copy()

# %%
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# %%

# Q3:
# OPTION A: with 0
X_train_0 = X_train.copy()
X_train_0["horsepower"] = X_train["horsepower"].fillna(0)

X_val_0 = X_val.copy()
X_val_0["horsepower"] = X_val["horsepower"].fillna(0)

model = linear_model.LinearRegression()
model.fit(X_train_0, y_train)

y_val_predict = model.predict(X_val_0)

rmse_val = metrics.root_mean_squared_error(y_val, y_val_predict)

print("RMSE Validation:", round(rmse_val, 3))
# %%

# OPTION B: with MEAN
mean = X_train["horsepower"].mean()

X_train_mean = X_train.copy()
X_train_mean["horsepower"] = X_train["horsepower"].fillna(mean)

X_val_mean = X_val.copy()
X_val_mean["horsepower"] = X_val["horsepower"].fillna(mean)

model = linear_model.LinearRegression()
model.fit(X_train_mean, y_train)

y_val_predict = model.predict(X_val_mean)

rmse_val = metrics.root_mean_squared_error(y_val, y_val_predict)

print("RMSE Validation:", round(rmse_val, 3))

# %%
# Q4:
r_values = [0, 0.01, 0.1, 1, 5, 10, 100]

for r in r_values:

    X_train_0 = X_train.copy()
    X_train_0["horsepower"] = X_train["horsepower"].fillna(0)

    X_val_0 = X_val.copy()
    X_val_0["horsepower"] = X_val["horsepower"].fillna(0)

    model = linear_model.Ridge(alpha=r)
    model.fit(X_train_0, y_train)

    y_val_predict = model.predict(X_val_0)

    rmse_val = metrics.root_mean_squared_error(y_val, y_val_predict)

    print(r, round(rmse_val, 4))

# %%
# Q5:
seed_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
seed_rmse = []

for s in seed_values:

    np.random.seed(s)
    idx = np.arange(n)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_train + n_val]]
    df_test = df.iloc[idx[n_train + n_val:]]

    X_train = df_train[features].copy()
    X_val = df_val[features].copy()

    y_train = df_train[target]
    y_val = df_val[target]

    X_train["horsepower"] = X_train["horsepower"].fillna(0)
    X_val["horsepower"] = X_val["horsepower"].fillna(0)

    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)

    y_val_predict = model.predict(X_val)

    rmse_val = metrics.root_mean_squared_error(y_val, y_val_predict)

    print(f"Seed: {s} | RMSE Validation: {round(rmse_val, 4)}")
    seed_rmse.append(rmse_val)

round(np.std(seed_rmse),3)
# %%
# Q6:
np.random.seed(9)
idx = np.arange(n)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val:]]

X_train = df_train[features].copy()
X_val = df_val[features].copy()
X_test = df_test[features].copy()

y_train = df_train[target]
y_val = df_val[target]
y_test = df_test[target]

X_train["horsepower"] = X_train["horsepower"].fillna(0)
X_val["horsepower"] = X_val["horsepower"].fillna(0)
X_test["horsepower"] = X_test["horsepower"].fillna(0)

X_full_train = pd.concat([X_train, X_val])
y_full_train = pd.concat([y_train, y_val])

model = linear_model.Ridge(alpha=0.001)
model.fit(X_full_train, y_full_train)

y_test_predict = model.predict(X_test)

rmse_test = metrics.root_mean_squared_error(y_test, y_test_predict)

print(f"RMSE Test: {round(rmse_test, 4)}")