# %load q01_bagging/build.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
plt.switch_backend('agg')

# Data Loading
dataframe = pd.read_csv('data/loan_prediction.csv')

X = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)
def bagging(X_train, X_test, y_train, y_test,n_est):
    score_dt =[]#create empty list to store scores for different value of estimators
    score_dt1 =[]
    array1=[10,20,30,40,50]
    for i in [0,1,2,3,4]:
        bagging_clf2 = BaggingClassifier(DecisionTreeClassifier(),n_estimators=array1[i],max_samples=0.67,max_features=0.67,bootstrap=True, random_state=9)
        bagging_clf2.fit(X_train, y_train)
        y_pred_decision=bagging_clf2.predict(X_test)
        score_dt.append(accuracy_score(y_test, y_pred_decision))
        y_pred_decision1=bagging_clf2.predict(X_train)
        score_dt1.append(accuracy_score(y_train, y_pred_decision1))
        plt.plot(array1[i],score_dt[i])
        plt.plot(array1[i],score_dt1[i])
        plt.show()


