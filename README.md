# Arrhythmia Detection using Deep Learning (CNN + LSTM + Autoencoder)

## Project Overview
This project focuses on automated arrhythmia (irregular heartbeat) detection from ECG signals using deep learning architectures.  
Two complementary models are implemented:

1. CNN + LSTM (Supervised Classification) – detects arrhythmia types from labeled ECG signals.  
2. CNN Autoencoder (Unsupervised Anomaly Detection) – identifies abnormal ECG beats without explicit labels.

Both models are trained and evaluated on the MIT-BIH Arrhythmia Dataset, a benchmark in ECG-based cardiac diagnostics.

---

## Motivation
Cardiovascular diseases remain one of the leading causes of death worldwide.  
Early detection of arrhythmias plays a crucial role in preventing severe cardiac conditions.  
This project demonstrates how deep learning can learn ECG patterns to support medical professionals in early and accurate diagnosis.

---

## Methodology

### 1. Data Preprocessing
- Dataset: MIT-BIH Arrhythmia Database (available on PhysioNet)
- Normalization using StandardScaler
- ECG segments reshaped to (samples, timesteps, 1) for 1D convolution input

---

### 2. CNN + LSTM Architecture (Supervised)
The CNN extracts morphological ECG features, while LSTM learns temporal dependencies between heartbeats.

**Architecture**
| Layer | Description |
|:------|:-------------|
| Conv1D | Extracts local heartbeat features |
| MaxPooling1D | Reduces dimensionality |
| LSTM | Captures sequence and rhythm |
| Dense | Classifies into arrhythmia categories |

**Loss:** categorical_crossentropy  
**Optimizer:** Adam  
**Metric:** Accuracy

---

### 3. CNN Autoencoder (Unsupervised Anomaly Detection)
Trained on normal beats only, the autoencoder reconstructs normal ECGs.  
Abnormal signals yield higher reconstruction errors, marking them as anomalies.

**Architecture**
| Encoder | Decoder |
|:--------|:--------|
| Conv1D → MaxPooling | UpSampling → Conv1D |
| Filters: 128 → 64 → 32 → 16 | Sigmoid reconstruction |

**Loss:** Mean Squared Error (MSE)  
**Metric:** Reconstruction Error Threshold (mean)

---

## Results

| Model | Task | Accuracy | Loss | Type |
|:------|:------|:---------|:------|:------|
| CNN + LSTM | Arrhythmia Classification | ~97% | 0.23 | Supervised |
| CNN Autoencoder | Anomaly Detection | — | 0.002–0.01 (MSE) | Unsupervised |

CNN + LSTM achieved robust classification accuracy.  
The Autoencoder effectively distinguished normal vs. abnormal ECG signals.

---

## Tech Stack
- Language: Python  
- Framework: TensorFlow / Keras  
- Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn  
- Tools: Jupyter Notebook / Google Colab  

---

## Repository Structure
