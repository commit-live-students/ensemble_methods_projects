import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np

# Data Loading
dataframe = pd.read_csv('data/loan_prediction.csv')

X = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)
a=[1,2,3,4,5,6,7,8,9,
  10,1,12,13,14,15,16,17,18,19,20,
  21,22,23,24,25,26,27,28,29,30,
  31,32,33,34,35,36,37,38,39,40,
  41,42,43,44,45,46,47,48,49,50]

def bagging(X_train,X_test,y_train,y_test,n_est):


    score_train=[]
    score_test=[]



    for i in a :
        decision_clf = DecisionTreeClassifier()
        bagging_clf1 = BaggingClassifier(DecisionTreeClassifier(), n_estimators=i, max_samples=0.67,
                                bootstrap=True, random_state=9)
        bagging_clf1.fit(X_train, y_train)
        y_pred_bagging = bagging_clf1.predict(X_test)
        score_bc_dt = accuracy_score(y_test, y_pred_bagging)
        score_test.append(score_bc_dt)
        y_pred_train_bagging = bagging_clf1.predict(X_train)
        score_bc_dtrain = accuracy_score(y_train, y_pred_train_bagging)
        score_train.append(score_bc_dtrain)


    plt.plot(a,score_train)
    plt.plot(a,score_test)
    plt.show()
