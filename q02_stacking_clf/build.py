# %load q02_stacking_clf/build.py
# Default imports
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

dataframe = pd.read_csv('data/loan_prediction.csv')
X = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

clf1 = LogisticRegression(random_state=9)
clf2 = DecisionTreeClassifier(random_state=9)
clf3 = DecisionTreeClassifier(max_depth=9, random_state=9)

bagging_clf1 = BaggingClassifier(clf2, n_estimators=100, max_samples=100,
                                 bootstrap=True, random_state=9, oob_score=True)
bagging_clf2 = BaggingClassifier(clf1, n_estimators=100, max_samples=100,
                                 bootstrap=True, random_state=9, oob_score=True)
bagging_clf3 = BaggingClassifier(clf3, n_estimators=100, max_samples=100,
                                 bootstrap=True, random_state=9, oob_score=True)
model = [bagging_clf1, bagging_clf2, bagging_clf3]
def stacking_clf(model, X_train, y_train, X_test, y_test):
    y_pred_train, y_pred_test=[], []
    for ind_model in model:
        ind_model.fit(X_train, y_train)
        y_pred_train.extend(ind_model.predict(X_train))
        y_pred_test.extend(ind_model.predict(X_test))
    X_test_new =  X_test.append(X_test.append(X_test))
    X_train_new =  X_train.append(X_train.append(X_train))
    clf1.fit(X_train_new, y_pred_train)
    y_pred_test_metaclf = clf1.predict(X_test)
    return accuracy_score(y_test, y_pred_test_metaclf)




