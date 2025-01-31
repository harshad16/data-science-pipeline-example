import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

standard_scaler = os.getenv("standard_scaler", False)

with open("iris_dataset") as f:
    df = pd.read_csv(f)
labels = df.pop('Labels')

scaler = StandardScaler() if standard_scaler else MinMaxScaler()

df = pd.DataFrame(scaler.fit_transform(df))
df['Labels'] = labels
with open("normalized_iris_dataset", 'w') as f:
    df.to_csv(f)