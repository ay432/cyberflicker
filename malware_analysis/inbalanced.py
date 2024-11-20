from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import scipy.sparse
import collections

X_train = scipy.sparse.load_npz("X_train.npz")
y_train = np.load("y_train.npy")
X_test = scipy.sparse.load_npz("X_test.npz")
y_test = np.load("y_test.npy")

# RandomForest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rfPred = rf.predict(X_test)
print(collections.Counter(rfPred))
print(balanced_accuracy_score(y_test, rfPred))

# Weighted
rfWeighted = RandomForestClassifier(n_estimators=100, class_weight='balanced')
rfWeighted.fit(X_train, y_train)
rfWeightedPred = rfWeighted.predict(X_test)
print(collections.Counter(rfWeightedPred))
print(balanced_accuracy_score(y_test, rfWeightedPred))

# Up sample
from sklearn.utils import resample
X_train_np = X_train.toarray()
class_0_indices = [i for i, x in enumerate(y_train==0) if x]
class_1_indices = [i for i, x in enumerate(y_train==1) if x]
size_class_0 = sum(y_train==0)
X_train_class_0 = X_train_np[class_0_indices,:]
y_train_class_0 = [0] * size_class_0
X_train_class_1 = X_train_np[class_1_indices,:]

X_train_class_1_resampled = resample(X_train_class_1, replace=True, n_samples=size_class_0)
y_train_class_1_resampled = [1] * size_class_0

X_train_resampled = np.concatenate([X_train_class_0, X_train_class_1_resampled])
y_train_resampled = y_train_class_0 + y_train_class_1_resampled

from scipy import sparse
X_train_class_1_resampled = sparse.csr_matrix(X_train_resampled)

rfResampled = RandomForestClassifier(n_estimators=100)
rfResampled.fit(X_train_resampled, y_train_resampled)
rfResampledPred = rfResampled.predict(X_test)
print(collections.Counter(rfResampledPred))
print(balanced_accuracy_score(y_test, rfResampledPred))

# Down sample
X_train_np = X_train.toarray()
class_0_indices = [i for i, x in enumerate(y_train==0) if x]
class_1_indices = [i for i, x in enumerate(y_train==1) if x]
size_class_1 = sum(y_train==1)
X_train_class_1 = X_train_np[class_0_indices,:]
y_train_class_1 = [0] * size_class_1
X_train_class_0 = X_train_np[class_1_indices,:]

X_train_class_0_downsampled = resample(X_train_class_0, replace=True, n_samples=size_class_1)
y_train_class_0_downsampled = [0] * size_class_1

X_train_downsampled = np.concatenate([X_train_class_1, X_train_class_0_downsampled])
y_train_downsampled = y_train_class_1 + y_train_class_0_downsampled

X_train_downsampled = sparse.csr_matrix(X_train_downsampled)

rfDownsampled = RandomForestClassifier(n_estimators=100)
rfDownsampled.fit(X_train_downsampled, y_train_downsampled)
rfDownsampledPred = rfDownsampled.predict(X_test)
print(collections.Counter(rfDownsampledPred))
print(balanced_accuracy_score(y_test, rfDownsampledPred))

# Classifier for unbalanced dataset
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier

BBC = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(), sampling_strategy='auto', replacement=False)
BBC.fit(X_train, y_train)
BBCPred = BBC.predict(X_test)
print(collections.Counter(BBCPred))
print(balanced_accuracy_score(y_test, BBCPred))

