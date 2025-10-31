import pandas as pd
import keras
from keras import layers
from keras import Model
import keras_tuner

features = [
    "speed",
    "temperature",
    "density",
    "bz_gsm",
    "bx_gsm",
    "by_gsm",
    "pressure",
    "smoothed_ssn",
]

x = pd.read_csv("x.csv")
y = pd.read_csv("train/dst_labels.csv")
stats = pd.read_csv("stats.csv")
a, b = stats.iloc[-1]

# a
x1 = x[x["period"] == "train_a"][["timedelta"] + features]
x1.reset_index(inplace=True, drop=True)
x1["timedelta"] = pd.to_timedelta(x1["timedelta"])

x1 = x1.set_index("timedelta").resample("h").agg(["mean", "std"])
x1.columns = ["_".join(col) for col in x1]
x1.drop("smoothed_ssn_std", axis=1, inplace=True)

y1 = y[y["period"] == "train_a"]["dst"]
y1 = (y1 - a) / b

length = len(x1.columns)  # length for model.fit

win = 24 * 7
# validation split
# period a
v_len = int(round(len(x1) * 0.2))  # val size is 20%
ind1 = len(x1) - v_len  # index separating validation

xt1 = x1.iloc[:ind1]  # train
yt1 = y1.iloc[:ind1]
xv1 = x1.iloc[ind1:]  # validate
yv1 = y1.iloc[ind1:]

# convert from dataframe to array
xt1 = xt1.values
yt1 = yt1.values

xv1 = xv1.values
yv1 = yv1.values
t = 1
# sliding windows
batch = 32
train_1 = keras.utils.timeseries_dataset_from_array(
    data=xt1,
    targets=yt1[win + t :],
    sequence_length=win,
    sampling_rate=1,
    sequence_stride=1,
    batch_size=batch,
)

val_1 = keras.utils.timeseries_dataset_from_array(
    data=xv1,
    targets=yv1[win + t :],
    sequence_length=win,
    sampling_rate=1,
    sequence_stride=1,
    batch_size=batch,
)

yv = yv1[win + t :] * b + a


# model
def build_model(hp):
    F = hp.Int("Filters", min_value=20, max_value=100, step=20)
    L = hp.Float("Learning_Rate", min_value=0.0001, max_value=0.001, step=0.0001)
    K = hp.Int("Kernel_Size", min_value=1, max_value=10, step=1)
    S = hp.Int("Stride", min_value=1, max_value=2, step=1)

    inputs = layers.Input(shape=(win, length))

    x = layers.Conv1D(F, kernel_size=K, strides=S, activation="relu")(inputs)
    x = layers.Conv1D(F, kernel_size=K, strides=S, activation="relu")(x)
    x = layers.Conv1D(2 * F, kernel_size=2 * K, strides=2 * S, activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(2 * F, activation="relu")(x)
    outputs = layers.Dense(1, activation="linear")(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=L),
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    return model


# Define the tuner
tuner = keras_tuner.RandomSearch(
    build_model,
    objective="val_loss",
    executions_per_trial=3,
    directory="final_h",
    project_name="run1",
)

# Early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

# Run hyperparameter search
tuner.search(
    train_1, validation_data=val_1, epochs=20, callbacks=[early_stop], verbose=2
)
