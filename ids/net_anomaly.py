import pandas as pd
df = pd.read_csv("kddcup.csv", index_col=None)
df.head()
df.shape
# Dataset is unbalanced and some attacks occurring only ones
labels = df["label"].values
from collections import Counter
Counter(labels)

def normalOrAnomalous(x):
    if x == "normal":
        return 0
    else:
        return 1

df["label"] = df["label"].map(normalOrAnomalous)
df["label"] = pd.to_numeric(df["label"])

from sklearn.preprocessing import LabelEncoder
encoding = dict()
for c in df.columns:
    if df[c].dtype == "object":
        encoding[c] = LabelEncoder
        df[c] = encoding[c].fit_transform(df[c])

y_normal = df_normal.pop("labels".values)
X_normal = df_normal.values
y_anomalous = df_anomalous.pop("labels".values)
X_anomalous = df_anomalous.vlaues

import numpy as np
from sklearn.model_selection import train_test_split

X_normal_train, X_normal_test, y_normal_train, y_normal_test = train_test_split(X_normal, y_normal, labels, test_size=0.02)
X_anomalous_train, X_anomalous_test, y_anomalous_train, y_anomalous_test = train_test_split(X_anomalous, y_anomalous, labels, test_size=0.02)

X_train = np.concatenate([X_normal_train, X_anomalous_train])
y_train = np.concatenate([y_normal_train, y_anomalous_train])
X_test = np.concatenate([X_normal_test, X_anomalous_test])
y_test = np.concatenate([y_normal_test, y_anomalous_test])

from sklearn.ensemble import IsolationForest
contaminationParameter = 1 - sum(y_train == 0)/len(y_train)
clf = IsolationForest(n_estimators=100, max_samples=256, contamination=contaminationParameter)

clf.fit(X_train)

import matplotlib.pyplot as plt
train_scores = clf.decision_function(X_normal_train)
fig = plt.figure(figsize=(8 ,4), dpi=600, facecolor='w', edgecolor='k')
normal = plt.hist(train_scores, 50, density=True)
plt.xlim(-0.2,0.2)
plt.xlabel('Anomaly Score')
plt.ylabel('Percentage')
plt.title("Distribution of anomaly scores for training normal set")

train_scores = clf.decision_function(X_anomalous_train)
fig = plt.figure(figsize=(8 ,4), dpi=600, facecolor='w', edgecolor='k')
normal = plt.hist(train_scores, 50, density=True)
plt.xlim(-0.2,0.2)
plt.xlabel('Anomaly Score')
plt.ylabel('Percentage')
plt.title("Distribution of anomaly scores for training anomalous set")

# Allows to set the cutoff
cutoff = 0.01

def FPR(y_true, y_pred):
    TotalNegatives = sum(y_true==0)
    FP = sum(y_pred[y_true==0]==1)
    return str(float(FP)/float(TotalNegatives)*100)+"%"

def TPR(y_true, y_pred):
    TotalPositives = sum(y_true == 1)
    TP = sum(y_pred[y_true == 1] == 1)
    return str(float(TP) / float(TotalPositives) * 100) + "%"

print(TPR(y_train, (cutoff>clf.decision_function(X_train)).astype(int)))
print(FPR(y_train, (cutoff>clf.decision_function(X_train)).astype(int)))

print(TPR(y_test, (cutoff>clf.decision_function(X_test)).astype(int)))
print(FPR(y_test, (cutoff>clf.decision_function(X_test)).astype(int)))
