# %load q02_stacking_clf/build.py
# Default imports
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

X_train, y_train,X_test, y_test = train_test_split(X, y, test_size=0.3,random_state=9)

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

#Actual function call
dataframe = pd.read_csv('data/loan_prediction.csv')
X = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1]

X_train,X_test,y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=9)
print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)

def stacking_clf(model, X_train,y_train, X_test, y_test):

    x_train_mdl = pd.DataFrame()
    for mdl in model:

        mdl.fit(X_train, y_train)
        x_train_mdl = pd.concat( [x_train_mdl, pd.DataFrame( mdl.predict_proba(X_train))]
                                ,axis=1)

    mdl_clf = LogisticRegression(random_state=9)
    mdl_clf.fit(x_train_mdl,y_train)

    x_test_mdl= pd.DataFrame()
    for mdl in model:
        x_test_mdl = pd.concat( [x_test_mdl, pd.DataFrame( mdl.predict_proba(X_test))]
                                ,axis=1)

    y_pred = mdl_clf.predict(x_test_mdl)

    score = accuracy_score(y_test, y_pred)
    return float(score)
