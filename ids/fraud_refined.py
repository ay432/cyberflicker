import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Function to generate the dataset
def generate_cost_sensitive_dataset(n_samples=1000, n_features=10, fraud_ratio=0.05, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    X = np.random.rand(n_samples, n_features)
    y = np.random.choice([0, 1], size=n_samples, p=[1 - fraud_ratio, fraud_ratio])
    amount = np.random.exponential(scale=50, size=n_samples)
    amount[y == 1] *= 5
    columns = [f"Feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df["Amount"] = amount.round(2)
    df["Class"] = y
    return df

# Generate the dataset
df = generate_cost_sensitive_dataset(
    n_samples=1000,
    n_features=5,
    fraud_ratio=0.1,
    random_state=42
)

# Prepare features and labels
X = df.drop(columns=["Class"])
y = df["Class"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Print class distribution
print("Class distribution before resampling:")
print(y_train.value_counts())

# Custom class weights for Scikit-learn
class_weights = {
    0: 1,  # Lower weight for non-fraud
    1: 10  # Higher weight for fraud
}

# Train a classifier with custom class weights
clf_weighted = RandomForestClassifier(class_weight=class_weights, random_state=42)
clf_weighted.fit(X_train, y_train)

# Make predictions and evaluate
y_pred_weighted = clf_weighted.predict(X_test)
print("\nEvaluation with custom class weights:")
print(classification_report(y_test, y_pred_weighted))
print(f"Accuracy: {accuracy_score(y_test, y_pred_weighted):.2f}")

# Handling imbalance with SMOTE (imbalanced-learn)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Print class distribution after SMOTE
print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# Train a classifier on resampled data
clf_resampled = RandomForestClassifier(random_state=42)
clf_resampled.fit(X_train_resampled, y_train_resampled)

# Make predictions and evaluate
y_pred_resampled = clf_resampled.predict(X_test)
print("\nEvaluation with SMOTE resampling:")
print(classification_report(y_test, y_pred_resampled))
print(f"Accuracy: {accuracy_score(y_test, y_pred_resampled):.2f}")
