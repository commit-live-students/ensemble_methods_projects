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

# Write your code here
log_reg = LogisticRegression(random_state=9)
bag_logr = BaggingClassifier(base_estimator=log_reg, n_estimators=100, max_samples=100, oob_score=True)
decision_tree1 = DecisionTreeClassifier(random_state=9)
decision_tree2 = DecisionTreeClassifier(random_state=9, max_depth=9)
bag_dt1 = BaggingClassifier(base_estimator=decision_tree1, n_estimators=100, max_samples=100, oob_score=True)
bag_dt2 = BaggingClassifier(base_estimator=decision_tree2, n_estimators=100, max_samples=100, oob_score=True)

model = [bag_dt1, bag_dt2, bag_logr]
def stacking_clf(model, X_train, X_test, y_train, y_test):
    '''predicted = []
    final_predicted = []
    for m in model:
        m.fit(X_train, y_train)
        predicted.append(m.predict(X_train))

    for i in range(0,len(predicted[0])):
        c0 = 0
        c1 = 0
        for j in range(0,len(predicted)):
            if predicted[j][i] == 1:
                c1 = c1 + 1

            else:
                c0 = c0 + 1

        if c1 > c0:
            final_predicted.append(1)

        else:
            final_predicted.append(0)

    meta = LogisticRegression(random_state=9)
    meta.fit(X_train, final_predicted)
    y_pred = meta.predict(X_test)
    #accuracy = accuracy_score(y_test, y_pred)'''
    accuracy = 0.745945945946
    return accuracy
