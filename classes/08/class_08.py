# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow
from tensorflow import keras

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions

# %%
path = "./data/clothing-dataset-small/train/t-shirt"
name = "0a85a584-cb49-4795-b2f1-7eebbf09399a.jpg"

full_name = f"{path}/{name}"

# %%
img = load_img(full_name, target_size=(299,299))
print(img)

# %%
x = np.array(img)
x.shape

# %%


model = Xception(
    weights="imagenet",
    input_shape=(299, 299, 3)
)

# %%
X = np.array([x])
X.shape

# %%
X = preprocess_input(X)
X

# %%
pred = model.predict(X)
pred.shape

# %%
decode_predictions(pred)

# %%