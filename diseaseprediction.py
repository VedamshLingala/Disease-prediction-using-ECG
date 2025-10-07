# cnn_lstm_mitbih_classification.py
# Requirements:
#   pip install numpy pandas scikit-learn matplotlib seaborn tensorflow joblib

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import class_weight
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ---------------------- Config ----------------------
TRAIN_CSV = "mitbih_train.csv"
TEST_CSV  = "mitbih_test.csv"
MODEL_OUT = "cnn_lstm_mitbih.h5"
SCALER_OUT = "mitbih_scaler.pkl"
BATCH_SIZE = 128
EPOCHS = 40
RANDOM_SEED = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------- Load data ----------------------
train_df = pd.read_csv(TRAIN_CSV, header=None)
test_df  = pd.read_csv(TEST_CSV, header=None)

# features are all columns except last, label is last column
X_train = train_df.iloc[:, :-1].values.astype(np.float32)
y_train = train_df.iloc[:, -1].values.astype(int)
X_test  = test_df.iloc[:, :-1].values.astype(np.float32)
y_test  = test_df.iloc[:, -1].values.astype(int)

print("Train shape:", X_train.shape, "Train labels distribution:\n", pd.Series(y_train).value_counts().sort_index())
print("Test shape :", X_test.shape,  "Test labels distribution:\n",  pd.Series(y_test).value_counts().sort_index())

# ---------------------- Preprocess: scale & reshape ----------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# reshape for Conv1D+LSTM: (samples, timesteps, features=1)
X_train_scaled = np.expand_dims(X_train_scaled, axis=2)  # shape (N, 187, 1)
X_test_scaled  = np.expand_dims(X_test_scaled, axis=2)

timesteps = X_train_scaled.shape[1]
n_classes = len(np.unique(y_train))
print("Timesteps:", timesteps, "Num classes:", n_classes)

# save scaler for later
joblib.dump(scaler, SCALER_OUT)

# ---------------------- Compute class weights to help imbalance ----------------------
cls_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: w for i, w in enumerate(cls_weights)}
print("Class weights:", class_weights)

# ---------------------- Build CNN + LSTM model ----------------------
def build_cnn_lstm(input_shape, n_classes):
    inp = layers.Input(shape=input_shape)                       # (187,1)
    # Conv feature extractor (keep padding='same' so time dims are preserved until pooling)
    x = layers.Conv1D(64, kernel_size=7, activation='relu', padding='same')(inp)
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)     # reduces timesteps ~ /2
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)     # reduces timesteps ~ /4
    x = layers.Dropout(0.25)(x)

    # LSTM on the sequence of features
    x = layers.LSTM(64, return_sequences=False)(x)              # output (batch, 64)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    out = layers.Dense(n_classes, activation='softmax')(x)

    model = models.Model(inp, out)
    return model

model = build_cnn_lstm(input_shape=(timesteps, 1), n_classes=n_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---------------------- Callbacks ----------------------
es = callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
mc = callbacks.ModelCheckpoint(MODEL_OUT, monitor='val_loss', save_best_only=True, verbose=1)

# ---------------------- Train ----------------------
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[es, mc],
    verbose=2
)

# save final model (best already saved by checkpoint)
model.save(MODEL_OUT)

# ---------------------- Evaluate on test set ----------------------
y_pred_proba = model.predict(X_test_scaled, batch_size= BATCH_SIZE)
y_pred = np.argmax(y_pred_proba, axis=1)

acc = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Test)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# plot training curves
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(); plt.title('Loss')
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend(); plt.title('Accuracy')
plt.show()

print(f"Saved model to {os.path.abspath(MODEL_OUT)} and scaler to {os.path.abspath(SCALER_OUT)}")
