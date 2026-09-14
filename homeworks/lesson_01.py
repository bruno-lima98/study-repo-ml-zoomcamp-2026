# %%

import pandas as pd
import numpy as np

# %%

# Q1:
pd.__version__

# %%
url = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"

df = pd.read_csv(url)

df.head()
# %%

# Q2:
len(df)

# %%
# Q3

print(df["fuel_type"].nunique())
print(df["fuel_type"].unique())

# %%

# Q4:
df.isna().sum()

# %%

# Q5:
df[df["origin"] == "Asia"]["fuel_efficiency_mpg"].max()

# %%

# Q6:
median = df["horsepower"].median()
mode = df["horsepower"].mode()[0]

print("median:", median)
print("mode:", mode)

# %%

df["horsepower"] = df["horsepower"].fillna(mode)
median_2 = df["horsepower"].median()

print("median:", median_2)

# %%

print(f"Median change: {median} -> {median_2}.")

# %%

# Q7:
X = np.array(df[df["origin"] == "Asia"][["vehicle_weight", "model_year"]].head(7))
X

# %%
XTX = X.T.dot(X)
XTX

# %%
INV = np.linalg.inv(XTX)
INV

# %%
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
y

# %%
w = (INV.dot(X.T)).dot(y)
w

# %%
w[0] + w[1]