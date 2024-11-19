import collections

import numpy as np
from nltk import ngrams

file = "C:\\Users\\jafar\\Downloads\\PEview\\PEview.exe"

def readFile(filePath):
    with open(filePath, "rb") as binary_file:
        data = binary_file.read()
    return data

def byteSequenceToNgrams(byteSequence, n):
    Ngrams = ngrams(byteSequence, n)
    return list(Ngrams)

def extractNgramsCounts(file, N):
    fileByteSequence = readFile(file)
    fileNgrams = byteSequenceToNgrams(fileByteSequence, N)
    return collections.Counter(fileNgrams)

# Extract 3-byte n-grams and count them
extractedNgrams = extractNgramsCounts(file, 3)

# Display the n-grams and their counts
for ng, count in extractedNgrams.items():
    print(f"{ng}: {count}")


# Feature selection
from os import listdir
from os.path import isfile, join

dirs = ["BenignSamples", "MaliciousSamples"]
N=2
totalNgramCount = collections.Counter([])

# Most frequent
for datasetPath in dirs:
    samples = [f for f in listdir(datasetPath) if isfile(join(datasetPath,f))]
    for file in samples:
        filePath = join(datasetPath, file)
        totalNgramCount += extractNgramsCounts(filePath, N)

K1 = 1000
K1_most_common = totalNgramCount.most_common(K1)
K1_most_common_list = [x[0] for x in K1_most_common]
print(K1_most_common_list)

def featurizeSample(file, K1_most_common_list):
    K1 = len(K1_most_common_list)
    fv = K1 * 0
    fileNgrams = extractNgramsCounts(file, N)
    for i in range(K1):
        fv[i] = fileNgrams[K1_most_common_list[i]]
    return fv

directoriesWithLabels = [("BenignSamples", 0), ("MaliciousSamples", 2)]
X = []
y = []
fileNum = 0
for datasetPath, label in directoriesWithLabels:
    samples = [f for f in listdir(datasetPath) if isfile(join(datasetPath, f))]
    for file in samples:
        fileNum += 1
        filePath = join(datasetPath, file)
        X.append(featurizeSample(filePath, K1_most_common_list))
        y.append(label)

X = np.asarray(X)

from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2

K2 = 100
X_top_k2_freq = X[:,:K2]

mi_selector = SelectKBest(mutual_info_classif, k=K2)
X_top_k2_mi = mi_selector.fit_transform(X, y)

chi2_selector = SelectKBest(chi2, k=K2)
X_top_k2_ch2 = chi2_selector.fit_transform(X, y)
# Mutual information
# Chi-squared
