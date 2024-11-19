# Thresholding, does the probability exceed threshold
# Pick higher probability

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import numpy as np
import scipy.sparse
import collections

X_train = scipy.sparse.load_npz("X_train.npz")
y_train = np.load("y_train.npy")
X_test = scipy.sparse.load_npz("X_test.npz")
y_test = np.load("y_test.npy")

desiredFPR = 0.05

def FPR(y_true, y_pred):
    CM = confusion_matrix(y_true, y_pred)
    TN = CM[0][0]
    FP = CM[0][1]
    FPR = FP/(FP + TN)
    return FPR

# Malware coverage
def TPR(y_true, y_pred):
    CM = confusion_matrix(y_true, y_pred)
    TP = CM[1][1]
    FN = CM[1][0]
    TPR = TP/(TP + FN)
    return TPR

def thresholdVector(vector, threshold):
    return [0 if x>=threshold else 1 for x in vector]

LR = LogisticRegression()
LR.fit(X_train, y_train)
LRPredProb = LR.predict_proba(X_train)
print("Probabilities look like so:")
print(LRPredProb[0:5])
print()
M = 100
print("Testing thresholds")

for threshold in reversed(range(M)):
    thresholdScaled = float(threshold)/M
    thresholdPrediction = thresholdVector(LRPredProb[:,0], thresholdScaled)
    print(threshold, FPR(y_train, thresholdPrediction), TPR(y_train, thresholdPrediction))
    if FPR(y_train, thresholdPrediction)<desiredFPR:
        print("Selected threshold: ")
        print(thresholdScaled)
        break

# Choose best threshold 0.62
