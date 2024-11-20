# Spam Filtering
import os

from nltk import accuracy
from sklearn.metrics import accuracy_score

spamPath = os.path.join("spamassassin-public-corpus", "spam")
hamPath = os.path.join("spamassassin-public-corpus", "ham")

corpus = []
labels = []

file_types_and_labels = [(spamPath, 0), (hamPath, 1)]

for filesPath, label in file_types_and_labels:
    files = os.listdir(filesPath)
    for file in files:
        file_path = filePath = os.path.join(filesPath, file)

        try:
            with open(file_path, "r") as myfile:
                data = myfile.read().replace('\n', '')
            data = str(data)
            corpus.append(data)
            labels.append(label)
        except:
            pass

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.33, random_state=42)

from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect',
                                HashingVectorizer(input='content', ngram_range=(1 ,3))),
                               ('tfidf', TfidfTransformer(use_idf=True, )), ('rf', RandomForestClassifier(class_weight='balanced')),])
text_clf.fit(X_train, y_train)

print(text_clf.score(X_train, y_train))

from sklearn.metrics import accuracy_score, confusion_matrix
y_test_pred = text_clf.predict(X_test)
print(accuracy_score(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))
