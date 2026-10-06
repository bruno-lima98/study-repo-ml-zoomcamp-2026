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
def make_model(learning_rate=0.01):
    base_model = Xception(
        weights="imagenet", 
        include_top=False,
        input_shape=(150, 150, 3)
        )

    base_model.trainable = False

    ##################################################################

    inputs = keras.Input(shape=(150, 150, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    outputs = keras.layers.Dense(10)(vectors)
    model = keras.Model(inputs, outputs)

    ##################################################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer, 
        loss=loss, 
        metrics=["accuracy"]
        )

    return model

# %%
learning_rate_values = [0.0001, 0.001, 0.01, 0.1]
scores = {}

for i in learning_rate_values:
    print(i)
    model = make_model(i)

    history = model.fit(
        train_ds, 
        epochs=10, 
        validation_data=val_ds
        )
    scores[i] = history.history

    print
    print()

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for lr, hist in scores.items():
    axes[0].plot(hist["accuracy"], label=lr)
    axes[1].plot(hist["val_accuracy"], label=lr)

axes[0].set_title("Training Accuracy")
axes[1].set_title("Validation Accuracy")

axes[0].set_xticks(np.arange(10))
axes[1].set_xticks(np.arange(10))

axes[0].set_xlabel("Epoch")
axes[1].set_xlabel("Epoch")

axes[0].set_ylabel("Accuracy")
axes[1].set_ylabel("Accuracy")

axes[0].legend()
axes[1].legend()

plt.tight_layout()
plt.show()

# %%
learning_rate = 0.001

# %%
model.save_weights("model_v1.weights.h5")

# %%
checkpoint = keras.callbacks.ModelCheckpoint(
    "xception_v1_{epoch:02d}_{val_accuracy:.3f}.h5",
    save_best_only=True,
    monitor="val_accuracy",
    mode="max"
    )

# %%
learning_rate = 0.001

model = make_model(learning_rate=learning_rate)

history = model.fit(
    train_ds, 
    epochs=10, 
    validation_data=val_ds,
    callbacks=[checkpoint]
    )

# %%
def make_model(learning_rate=0.01, size_inner=100):
    base_model = Xception(
        weights="imagenet", 
        include_top=False,
        input_shape=(150, 150, 3)
        )

    base_model.trainable = False

    ##################################################################

    inputs = keras.Input(shape=(150, 150, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)

    inner = keras.layers.Dense(size_inner, activation="relu")(vectors)

    outputs = keras.layers.Dense(10)(inner)
    model = keras.Model(inputs, outputs)

    ##################################################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer, 
        loss=loss, 
        metrics=["accuracy"]
        )

    return model

# %%
learning_rate = 0.001

size_inner_values = [10, 100, 1000]
scores = {}

for i in size_inner_values:
    print(i)
    model = make_model(
        learning_rate=learning_rate,
        size_inner=i
    )

    history = model.fit(
        train_ds, 
        epochs=10, 
        validation_data=val_ds
        )
    scores[i] = history.history

    print
    print()

# %%
for size, hist in scores.items():
    plt.plot(hist["val_accuracy"], label=(f"size_inner={size}"))

plt.xticks(np.arange(10))
plt.legend()
plt.show()

# %%
model.save_weights("model_v2.weights.h5")
model.save("model_v2.keras")

# %%
def make_model(learning_rate=0.01, size_inner=100, droprate=0.5):
    base_model = Xception(
        weights="imagenet", 
        include_top=False,
        input_shape=(150, 150, 3)
        )

    base_model.trainable = False

    ##################################################################

    inputs = keras.Input(shape=(150, 150, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)

    inner = keras.layers.Dense(size_inner, activation="relu")(vectors)
    drop = keras.layers.Dropout(droprate)(inner)

    outputs = keras.layers.Dense(10)(drop)
    model = keras.Model(inputs, outputs)

    ##################################################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer, 
        loss=loss, 
        metrics=["accuracy"]
        )

    return model

# %%
learning_rate = 0.001
size = 100

dropout_values = [0.0, 0.2, 0.5, 0.8]
scores = {}

for i in dropout_values:
    print(i)
    model = make_model(
        learning_rate=learning_rate,
        size_inner=size,
        droprate=i
    )

    history = model.fit(
        train_ds, 
        epochs=10, 
        validation_data=val_ds
        )
    scores[i] = history.history

    print
    print()

