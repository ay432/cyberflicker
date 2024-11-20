# Cost Sensitive Classification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from costcla.models import CostSensitiveRandomForestClassifier
from costcla.metrics import savings_score

df = pd.read_csv("creditcard.csv", index_col=None)

df.head()

from collections import Counter
Counter(df["Class"].values)

# Administrative cost if they get classification wrong
admin_cost = 5

cost_mat = np.zeros((len(df.index), 4))
cost_mat[:,0] = admin_cost * np.ones(len(df.index))
cost_mat[:,1] = df["Amount"].values
cost_mat[:,2] = admin_cost * np.ones(len(df.index))

# Very expensive mistake if fraudulent transaction missed
print(cost_mat)

y =  df.pop("Class").values
X=df.values

sets = train_test_split(X, y, cost_mat, test_size=0.2, random_state=0)
X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets

y_pred_test_rf = RandomForestClassifier(random_state=0).fit(X_train, y_train).predict(X_test)

y_pred_test_csrf = CostSensitiveRandomForestClassifier().fit(X_train, y_train, cost_mat_train).predict(X_test)

print(savings_score(y_test, y_pred_test_rf, cost_mat_test))
print(savings_score(y_test, y_pred_test_csrf, cost_mat_test))
# When data have its own cost metrics