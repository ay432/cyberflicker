import numpy as np
rng = np.random.RandomState(42)
nNormal = 2000
nAnomalous = 200
X_normal = np.r_[0.5 * rng.rand(int(nNormal/2), 2), 0.5 * rng.rand(int(nNormal/2), 2)]
y_normal = nNormal * [0]
X_anomalous = rng.uniform(low=-5, high=5, size=(nAnomalous, 2))
y_anomalous = nAnomalous*[1]

print(X_normal.shape)
print(X_normal[0:5])

print(X_anomalous.shape)
print(X_anomalous[0:5])

import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [15, 10]
plt.title("Data")

p1 = plt.scatter(X_normal[:,0], X_normal[:,1], c='white', s=20*4, edgecolor='k')
p2 = plt.scatter(X_anomalous[:,0], X_anomalous[:,1], c='green', s=20*4, edgecolor='k')

plt.axis('tight')
plt.xlim((-6 ,6))
plt.ylim((-6 ,6))
plt.legend([p1 ,p2],
           ["Normal observations",
            "Anomalous observations"],
           loc="lower right")

# plt.show()

from sklearn.model_selection import train_test_split

X_normal_train, X_normal_test, y_normal_train, y_normal_test = train_test_split(X_normal, y_normal, test_size=0.2)

X_anomalous_train, X_anomalous_test, y_anomalous_train, y_anomalous_test = train_test_split(X_anomalous, y_anomalous, test_size=0.2)
X_train = np.concatenate([X_normal_train, X_anomalous_train])
y_train = np.concatenate([y_normal_train, y_anomalous_train])
X_test = np.concatenate([X_normal_test, X_anomalous_test])
y_test = np.concatenate([y_normal_test, y_anomalous_test])

# Print shapes to verify
print(f"Training set shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Testing set shape: {X_test.shape}, Testing labels shape: {y_test.shape}")

from sklearn.ensemble import IsolationForest
clf = IsolationForest()
clf.fit(X_train)

train_scores = clf.decision_function(X_normal_train)
fig = plt.figure(figsize=(8, 4), dpi=600, facecolor='w', edgecolor='k')
normal = plt.hist(train_scores, 50, density=True)
plt.xlim(-0.2, 0.2)
plt.xlabel('Anomaly score')
plt.ylabel('Percentage')
plt.title("Distribution of anomaly scores for training normal set")

train_scores = clf.decision_function(X_anomalous_train)
fig = plt.figure(figsize=(8, 4), dpi=600, facecolor='w', edgecolor='k')
normal = plt.hist(train_scores, 50, density=True)
plt.xlim(-0.2, 0.2)
plt.xlabel('Anomaly score')
plt.ylabel('Percentage')
plt.title("Distribution of anomaly scores for training anomalous set")

#plt.show()

cutoff = 0.01

def FPR(y_true, y_pred):
    TotalNegatives = sum(y_true==0)
    FP = sum(y_pred[y_true==0]==1)
    return str(float(FP)/float(TotalNegatives)*100)+"%"

def TPR(y_true, y_pred):
    TotalPositives = sum(y_true==1)
    TP = sum(y_pred[y_true==1]==1)
    return str(float(TP)/float(TotalPositives)*100)+"%"

print(TPR(y_train, (cutoff>clf.decision_function(X_train)).astype(int)))
print(FPR(y_train, (cutoff>clf.decision_function(X_train)).astype(int)))

print(TPR(y_test, (cutoff>clf.decision_function(X_test)).astype(int)))
print(FPR(y_test, (cutoff>clf.decision_function(X_test)).astype(int)))

