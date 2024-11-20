import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# Generate data
rng = np.random.default_rng(seed=42)
n_normal = 2000
n_anomalous = 200

X_normal = np.vstack([
    0.5 * rng.random(size=(n_normal // 2, 2)),
    0.5 * rng.random(size=(n_normal // 2, 2))
])
y_normal = np.zeros(n_normal, dtype=int)

X_anomalous = rng.uniform(low=-5, high=5, size=(n_anomalous, 2))
y_anomalous = np.ones(n_anomalous, dtype=int)

# Split data
X_normal_train, X_normal_test, y_normal_train, y_normal_test = train_test_split(X_normal, y_normal, test_size=0.2, random_state=42)
X_anomalous_train, X_anomalous_test, y_anomalous_train, y_anomalous_test = train_test_split(X_anomalous, y_anomalous, test_size=0.2, random_state=42)

X_train = np.vstack([X_normal_train, X_anomalous_train])
y_train = np.hstack([y_normal_train, y_anomalous_train])
X_test = np.vstack([X_normal_test, X_anomalous_test])
y_test = np.hstack([y_normal_test, y_anomalous_test])

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Isolation Forest
clf = IsolationForest(contamination=n_anomalous / (n_normal + n_anomalous), random_state=42)
clf.fit(X_train_scaled)

# Decision scores
train_scores = clf.decision_function(X_train_scaled)
test_scores = clf.decision_function(X_test_scaled)

# Optimize cutoff threshold using ROC curve
y_combined = np.hstack([y_normal, y_anomalous])
X_combined = np.vstack([X_normal, X_anomalous])
combined_scores = clf.decision_function(scaler.transform(X_combined))

fpr, tpr, thresholds = roc_curve(y_combined, -combined_scores)  # Negative for Isolation Forest
optimal_idx = np.argmax(tpr - fpr)
optimal_cutoff = thresholds[optimal_idx]
print(f"Optimal Cutoff: {optimal_cutoff:.4f}")

# Visualize decision scores
plt.hist(train_scores[:len(y_normal_train)], bins=50, alpha=0.6, label="Normal (Train)", color="blue", density=True)
plt.hist(train_scores[len(y_normal_train):], bins=50, alpha=0.6, label="Anomalous (Train)", color="red", density=True)
plt.axvline(optimal_cutoff, color="black", linestyle="--", label="Optimal Cutoff")
plt.xlabel("Anomaly Score")
plt.ylabel("Density")
plt.title("Anomaly Scores Distribution")
plt.legend()
plt.show()

# Evaluate model
def false_positive_rate(y_true, y_pred):
    total_negatives = np.sum(y_true == 0)
    false_positives = np.sum(y_pred[y_true == 0] == 1)
    return f"{(false_positives / total_negatives) * 100:.2f}%"

def true_positive_rate(y_true, y_pred):
    total_positives = np.sum(y_true == 1)
    true_positives = np.sum(y_pred[y_true == 1] == 1)
    return f"{(true_positives / total_positives) * 100:.2f}%"

# Predictions based on the optimal cutoff
y_train_pred = (clf.decision_function(X_train_scaled) < optimal_cutoff).astype(int)
y_test_pred = (clf.decision_function(X_test_scaled) < optimal_cutoff).astype(int)

# Print evaluation metrics
print("Training Data TPR:", true_positive_rate(y_train, y_train_pred))
print("Training Data FPR:", false_positive_rate(y_train, y_train_pred))
print("Testing Data TPR:", true_positive_rate(y_test, y_test_pred))
print("Testing Data FPR:", false_positive_rate(y_test, y_test_pred))