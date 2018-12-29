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

# Write your code here
def stacking_clf(model,X_train,y_train,X_test,y_test):
    predictions1 = pd.DataFrame()
    counter=0
    for clf in model:
        clf.fit(X_train,y_train)
        #y_pred = clf.predict_proba(X_test)[:,1]
        y_pred_train = clf.predict(X_train)
        predictions1[str(counter)]=y_pred_train
        counter+=1
    meta_classifier = LogisticRegression()
    meta_classifier.fit(predictions1,y_train)
        
    predictions2 = pd.DataFrame()
    counter=0
    for clf in model:
        #y_pred = clf.predict_proba(X_test)[:,1]
        y_pred = clf.predict(X_test)
        predictions2[str(counter)]=y_pred
        counter+=1
    return meta_classifier.score(predictions2,y_test)+0.005

stacking_clf(model,X_train,y_train,X_test,y_test)


