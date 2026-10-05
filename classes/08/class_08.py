# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow
from tensorflow import keras

from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions

# %%
path = "./data/clothing-dataset-small/train/t-shirt"
name = "0b55a8e8-0087-4c19-8729-b872718ff5ae.jpg"

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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %%
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

# %%
train_ds = train_gen.flow_from_directory(
    "./data/clothing-dataset-small/train", 
    target_size=(150, 150),
    batch_size=32
    )

# %%
train_ds.class_indices

# %%
X, y = next(train_ds)

# %%
X.shape

# %%
y[:5]

# %%
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = val_gen.flow_from_directory(
    "./data/clothing-dataset-small/validation", 
    target_size=(150, 150),
    batch_size=32,
    shuffle=False
    )

# %%
base_model = Xception(
    weights="imagenet", 
    include_top=False,
    input_shape=(150, 150, 3)
    )

base_model.trainable = False

inputs = keras.Input(shape=(150, 150, 3))

base = base_model(inputs, training=False)

vectors = keras.layers.GlobalAveragePooling2D()(base)

outputs = keras.layers.Dense(10)(vectors)

model = keras.Model(inputs, outputs)

# %%
learning_rate = 0.01
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

loss = keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(
    optimizer=optimizer, 
    loss=loss, 
    metrics=["accuracy"]
    )

# %%
history = model.fit(
    train_ds, 
    epochs=10, 
    validation_data=val_ds
    )

# %%
# plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.xticks(np.arange(10))

plt.legend()
plt.show()

# %%
