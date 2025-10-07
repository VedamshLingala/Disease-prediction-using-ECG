# ====================================================
# CNN Autoencoder for Anomaly Detection (Arrhythmia)
# ====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------- Step 1: Load Dataset ----------------------
data = pd.read_csv("/content/mitbih_train.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Use only *normal beats* for training (class 0)
X_normal = X[y == 0]
scaler = StandardScaler()
X_normal_scaled = scaler.fit_transform(X_normal)

# Reshape for CNN input
X_normal_scaled = X_normal_scaled.reshape((X_normal_scaled.shape[0], X_normal_scaled.shape[1], 1))

X_train, X_val = train_test_split(X_normal_scaled, test_size=0.2, random_state=42)

# ---------------------- Step 2: Build CNN Autoencoder ----------------------
input_shape = (X_train.shape[1], 1)
input_layer = Input(shape=input_shape)

# Encoder
x = Conv1D(64, 5, activation="relu", padding="same")(input_layer)
x = MaxPooling1D(2, padding="same")(x)
x = Conv1D(32, 3, activation="relu", padding="same")(x)
encoded = MaxPooling1D(2, padding="same")(x)

# Decoder
x = Conv1D(32, 3, activation="relu", padding="same")(encoded)
x = UpSampling1D(2)(x)
x = Conv1D(64, 5, activation="relu", padding="same")(x)
x = UpSampling1D(2)(x)
decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# ---------------------- Step 3: Train Autoencoder ----------------------
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

history = autoencoder.fit(
    X_train, X_train,
    epochs=30,
    batch_size=128,
    validation_data=(X_val, X_val),
    callbacks=[es],
    verbose=1
)

# ---------------------- Step 4: Plot Training Loss ----------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('CNN Autoencoder Training Loss')
plt.show()

# ---------------------- Step 5: Detect Anomalies ----------------------
X_all_scaled = scaler.transform(X)
X_all_scaled = X_all_scaled.reshape((X_all_scaled.shape[0], X_all_scaled.shape[1], 1))
reconstructions = autoencoder.predict(X_all_scaled)
mse = np.mean(np.power(X_all_scaled - reconstructions, 2), axis=(1,2))

# Compute threshold
threshold = np.percentile(mse, 95)
print(f"Anomaly Threshold: {threshold:.5f}")

y_pred = (mse > threshold).astype(int)

# Compare with actual labels
anomalies = np.sum(y_pred != (y == 0))
print(f"Detected anomalies:
