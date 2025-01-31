import os
import json
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import confusion_matrix


n_neighbors = int(os.getenv("n_neighbors", 3))

with open("normalized_iris_dataset") as f:
    df = pd.read_csv(f)

y = df.pop('Labels')
X = df

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = KNeighborsClassifier(n_neighbors=n_neighbors)
clf.fit(X_train, y_train)

predictions = cross_val_predict(
    clf, X_train, y_train, cv=3)
# metrics.log_confusion_matrix(
#     ['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica'],
#     confusion_matrix(
#         y_train,
#         predictions).tolist()  # .tolist() to convert np array to list.
# )

# with open('mlpipeline-metrics.json', 'w') as f:
#     json.dump(metrics, f)


with open("model", 'wb') as f:
    pickle.dump(clf, f)