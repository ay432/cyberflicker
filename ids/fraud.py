# Cost Sensitive Classification
import pandas as pd
df = pd.read_csv("creditcard.csv", index_col=None)

df.head()

from collections import Counter
Counter(df["Class"].values)

