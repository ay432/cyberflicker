from os import listdir
import os
import pefile
from os.path import isfile, join
from sklearn.model_selection import train_test_split
import collections
import numpy as np
from nltk import ngrams

directoriesWithLabels = [("BenignSamples", 0), ("MaliciousSamples", 1)]
listOfSamples = []
labels = []

for datasetPath, label in directoriesWithLabels:
    samples = [f for f in listdir(datasetPath)]
    for file in samples:
        filePath = os.path.join(datasetPath, file)
        listOfSamples.append(filePath)
        labels.append(label)

samples_train, samples_test, labels_train, labels_test = train_test_split(listOfSamples, labels, test_size=0.33,
                                                                          stratify=labels, random_state=42)

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

def getNGramfeaturesFromSample(file, K1_most_common_list):
    K1 = len(K1_most_common_list)
    fv = K1 * 0
    fileNgrams = extractNgramsCounts(file, N)
    for i in range(K1):
        fv[i] = fileNgrams[K1_most_common_list[i]]
    return fv

def preprocessImports(listOfDLLs):
    preprocessedListOfDLLs = []
    return [x.decode().split(".")[0].lower() for x in listOfDLLs]

def getImports(pe):
    listOfImports = []
    for entry in pe.DIRECTORY_ENTRY_IMPORT:
        listOfImports.append(entry.dll)
    return preprocessImports(listOfImports)

def getSectionNames(pe):
    listOfSectionNames = []
    for eachSection in pe.sections:
        refined_name = eachSection.Name.decode().replace('\x00','').lower()
        listOfSectionNames.append(refined_name)
    return listOfSectionNames

N=2
totalNgramCount = collections.Counter([])

for file in samples_train:
    totalNgramCount += extractNgramsCounts(file, N)

K1 = 100
K1_most_common = totalNgramCount.most_common(K1)
K1_most_common_list = [x[0] for x in K1_most_common]

importsCorpus_train = []
numSections_train = []
sectionNames_train = []
NgramFeaturesList_train = []
y_train = []

for i in range(len(samples_train)):
    filePath = samples_train[i]
    try:
        NGramFeatures = getNGramfeaturesFromSample(file, K1_most_common_list)
        pe = pefile.PE(filePath)
        imports = getImports(pe)
        nSections = len(pe.sections)
        secNames = getSectionNames(pe)
        importsCorpus_train.append(imports)
        numSections_train.append(nSections)
        sectionNames_train.append(secNames)
        NgramFeaturesList_train.append(NGramFeatures)
        y_train.append(labels_train[i])
    except Exception as e:
        print(file+":")
        print(e)

from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

imports_featurizer = Pipeline([('vect',
                                HashingVectorizer(input='content', ngram_range=(1 ,2))),
                               ('tfidf', TfidfTransformer(use_idf=True, )),])
section_names_featurizer = Pipeline([('vect',
                                      HashingVectorizer(input='content', ngram_range=(1 ,2))),
                                     ('tfidf', TfidfTransformer(use_idf=True, )),])

importsCorpus_train_transformed = imports_featurizer.fit_transform(importsCorpus_train)
section_names_train_transformed = section_names_featurizer.fit_transform(sectionNames_train)

from scipy.sparse import hstack, csr_matrix
X_train = hstack([NgramFeaturesList_train,
                  importsCorpus_train_transformed,
                  section_names_train_transformed,
                  csr_matrix(numSections_train).transpose])

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf = clf.fit(X_train, y_train)

clf.score(X_train, y_train)

importsCorpus_test = []
numSections_test = []
sectionNames_test = []
NgramFeaturesList_test = []
y_test = []

for i in range(len(samples_test)):
    filePath = samples_test[i]
    try:
        NGramFeatures = getNGramfeaturesFromSample(file, K1_most_common_list)
        pe = pefile.PE(filePath)
        imports = getImports(pe)
        nSections = len(pe.sections)
        secNames = getSectionNames(pe)
        importsCorpus_test.append(imports)
        numSections_test.append(nSections)
        sectionNames_test.append(secNames)
        NgramFeaturesList_test.append(NGramFeatures)
        y_test.append(labels_test[i])
    except Exception as e:
        print(file+":")
        print(e)

importsCorpus_test_transformed = imports_featurizer.transform(importsCorpus_test)
section_names_test_transformed = section_names_featurizer.fit_transform(sectionNames_test)

X_test = hstack([NgramFeaturesList_test,
                  importsCorpus_test_transformed,
                  section_names_test_transformed,
                  csr_matrix(numSections_test).transpose])

clf.score(X_test, y_test)

