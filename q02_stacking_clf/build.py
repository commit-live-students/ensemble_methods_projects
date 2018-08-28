# %load q02_stacking_clf/build.py
# Default imports
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from mlxtend.classifier import StackingClassifier
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier

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
def stacking_clf(model, X_train, y_train, X_test, y_test):
    voting_clf_hard = VotingClassifier(estimators = [('Logistic Regression', model[0]),
                                                 ('Decision Tree 1', model[1]),
                                                 ('Decision Tree 2', model[2])],
                                   voting = 'hard')

    voting_clf_hard.fit(X_train, y_train)
    y_pred_hard = voting_clf_hard.predict(X_train)
    y_pred_hard_test = voting_clf_hard.predict(X_test)
    X_trainN = np.concatenate((X_train,pd.DataFrame(y_pred_hard)), axis=1)
    X_testN = np.concatenate((X_test,pd.DataFrame(y_pred_hard_test)), axis=1)

    stacking_clf1 = StackingClassifier(classifiers = model,
                                 meta_classifier = LogisticRegression(random_state=9))
    stacking_clf1.fit(X_trainN, y_train)
    y_pred = stacking_clf1.predict(X_testN)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy



