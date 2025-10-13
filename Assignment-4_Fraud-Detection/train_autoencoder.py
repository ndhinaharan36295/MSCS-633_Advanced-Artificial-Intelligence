import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.auto_encoder import AutoEncoder
import joblib

# --- Configuration ---
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "creditcard.csv")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "model_autoencoder.joblib")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
RANDOM_STATE = 42
TEST_SIZE = 0.3

os.makedirs(FIG_DIR, exist_ok=True)

# --- Load data ---
print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# quick check for expected columns
if 'Class' not in df.columns:
    raise ValueError("Expected 'Class' column in dataset (0 = normal, 1 = fraud)")

# Optional: drop Time column if you don't want temporal info
if 'Time' in df.columns:
    df = df.drop(columns=['Time'])

# --- Prepare features and labels ---
X = df.drop(columns=['Class']).values
y = df['Class'].values  # 0 normal, 1 fraud

# Standardize features (important for neural nets)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train/test split (stratify to preserve fraud proportion) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# We'll train on normal-only subset for clearer anomaly signal.
X_train_norm = X_train[y_train == 0]

print(f"Training data: {X_train.shape[0]} rows (normal-only used: {X_train_norm.shape[0]} rows)")
print(f"Test data: {X_test.shape[0]} rows, with {y_test.sum()} frauds")

# --- Build AutoEncoder (PyOD) ---
ae = AutoEncoder(
    hidden_neurons=[64, 32, 32, 64],  # encoder/decoder structure
    hidden_activation='relu',
    output_activation='linear',
    loss='mse',
    epochs=50,
    batch_size=256,
    learning_rate=0.001,
    validation_size=0.1,
    verbose=1,
    random_state=RANDOM_STATE
)

# Fit model on normal-only data
ae.fit(X_train_norm)

# Save model + scaler
joblib.dump({'model': ae, 'scaler': scaler}, MODEL_OUT)
print("Saved model to:", MODEL_OUT)

# --- Get anomaly (reconstruction) scores ---
# pyod returns decision_scores_ (higher -> more abnormal)
test_scores = ae.decision_function(X_test)  # same as ae.decision_scores_ for training data

# Evaluate using ROC AUC (continuous score) and thresholded classification
roc_auc = roc_auc_score(y_test, test_scores)
print(f"ROC AUC on test set: {roc_auc:.4f}")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, test_scores)
roc_auc_val = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_val:.4f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - AutoEncoder Anomaly Scores")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "roc_curve.png"))
plt.close()

# We'll compute threshold = 99th percentile of training-normal decision scores.
train_scores = ae.decision_function(X_train_norm)
thr = np.percentile(train_scores, 99)  # change percentile if needed
print(f"Using threshold (99th percentile of training-normal scores): {thr:.6f}")

# Classify test points as anomaly (1) if score > thr
y_pred = (test_scores > thr).astype(int)

# Compute precision, recall, f1
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
print(f"Thresholded results -- Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# Also report confusion-style numbers
tp = int(((y_test == 1) & (y_pred == 1)).sum())
fp = int(((y_test == 0) & (y_pred == 1)).sum())
fn = int(((y_test == 1) & (y_pred == 0)).sum())
tn = int(((y_test == 0) & (y_pred == 0)).sum())
print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")

# --- Plot score distributions ---
plt.figure(figsize=(8,5))
sns.histplot(test_scores[y_test==0], label='Normal (test)', stat='density', kde=True, bins=100)
sns.histplot(test_scores[y_test==1], label='Fraud (test)', stat='density', kde=True, bins=100)
plt.axvline(thr, color='red', linestyle='--', label='Threshold')
plt.legend()
plt.xlabel("AutoEncoder anomaly score (higher = more abnormal)")
plt.title("Reconstruction scores: normal vs fraud (test)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "score_distribution.png"))
plt.close()

print("Saved figures to", FIG_DIR)
print("Done.")
