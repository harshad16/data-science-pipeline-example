import pandas as pd

csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
col_names = [
    'Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Labels'
]
df = pd.read_csv(csv_url, names=col_names)


with open("iris_dataset", 'w') as f:
    df.to_csv(f)