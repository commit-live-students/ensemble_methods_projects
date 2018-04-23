# Default imports
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Loading data
dataframe = pd.read_csv('data/loan_prediction.csv')
X = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

# Write your code here
def stacking_clf(model, X_train, y_train, X_test, y_test):
    for mod in model:
        mod.fit(X_train, y_train)
    X_train1 = X_test1 = pd.DataFrame()

    for mod in model:
        X_train1 = pd.concat([X_train1,pd.DataFrame(mod.predict_proba(X_train))],axis=1)

    for mod in model:
        X_test1 = pd.concat([X_test1,pd.DataFrame(mod.predict_proba(X_test))],axis=1)

    meta_clf = LogisticRegression(random_state=9)
    meta_clf.fit(X_train1,y_train)
    y_pred = meta_clf.predict(X_test1)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
