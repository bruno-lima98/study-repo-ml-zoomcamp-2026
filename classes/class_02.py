# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
# %%

df = pd.read_csv("data/data_class_02.csv")
df.head()

# %%
# 1. DATA PREPARATION

df.columns = df.columns.str.lower().str.replace(" ", "_").str.strip()

# %%

string_columns = list(df.dtypes[df.dtypes == "O"].index)

for column in string_columns:
    df[column] = df[column].str.lower().str.replace(" ", "_").str.strip()

# %% 
df.head()

# %%
# 2. EXPLORATION DATA ANALYSIS (EDA)

for col in df.columns:
    print(f"Column: {col}.")
    print(f"There are {df[col].nunique()} unique values.")
    print(df[col].unique()[:5])
    print()

# %%
sns.histplot(df[df["msrp"] <= 100000]["msrp"], bins=50)
plt.xlabel("Price")
plt.ylabel("Total")
plt.title("Price range of cars")
plt.show()

# Distribuição de longa cauda geralmente confunde o modelo

# %%
sns.histplot(np.log1p(df["msrp"]) , bins=50)
plt.xlabel("Price")
plt.ylabel("Total")
plt.title("Price range of cars")
plt.show()

# %%
df.isnull().sum()

# %%
missing_num = ["engine_hp", "engine_cylinders", "number_of_doors"]
missing_cat = ["market_category", "engine_fuel_type"]

for col in missing_num:
    mode = float(df[col].mode())
    df[col] = df[col].fillna(mode)

for col in missing_cat:
    df[col] = df[col].fillna("missing_info")

df.isnull().sum()

# %%
df["age"] = 2026 - df["year"]

# %%
# 3. SPLITING DATASET

from sklearn.model_selection import train_test_split

# %%

target = "msrp"
features = []
for i in df.columns:
    if i != target:
        features.append(i)

# %%
features
# %%
X = df[features]
y = df[target]

# %%
X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

# %%
print(len(df[features]))
print(len(df[target]))
print(len(X_train))
print(len(y_train))
print(len(X_val))
print(len(y_val))
print(len(X_test))
print(len(y_test))

# %%
X_train = X_train.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# %%
y_train = np.log1p(y_train)
y_val = np.log1p(y_train)
y_test = np.log1p(y_train)

# %%

columns = [
    "engine_hp",
    "engine_cylinders",
    "highway_mpg",
    "city_mpg",
    "popularity",
    "age",
    ]

X_train[columns].head()
# %%

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[columns])
X_val_scaled = scaler.transform(X_val[columns])
X_test_scaled = scaler.transform(X_test[columns])

# %%
model = linear_model.LinearRegression()
model.fit(X_train_scaled, y_train)

# %%
y_train_predict = model.predict(X_train_scaled)

# %%
rmse_train = metrics.root_mean_squared_error(
    y_train, y_train_predict
)

r2_train = metrics.r2_score(y_train, y_train_predict)

print("RMSE Treino:", rmse_train)
print("R² Treino:", r2_train)