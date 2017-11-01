# %load q01_bagging/build.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Data Loading
dataframe = pd.read_csv('data/loan_prediction.csv')

X = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)


# Write your code here

def bagging(X_train, X_test, y_train, y_test, n_est):
    #decision_clf = DecisionTreeClassifier()
    #decision_clf.fit(X_train, y_train)
    bagging_clf2 = BaggingClassifier(DecisionTreeClassifier(), n_estimators= n_est, max_samples=100,
                                bootstrap=True, random_state=9)

    bagging_clf2.fit(X_train, y_train)
    y_pred_bagging = bagging_clf2.predict(X_test)
    score_bc_dt = accuracy_score(y_test, y_pred_bagging)
    #return score_bc_dt
#bagging(X_train, X_test, y_train, y_test, 100)
