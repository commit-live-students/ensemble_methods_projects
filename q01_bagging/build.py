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
n_est = 50

# Write your code here
def bagging(X_train,X_test,y_train,y_test,n_est):
    dt_clf = DecisionTreeClassifier()
    acc_train = []
    acc_test = []
    list_no_estimators = []
    fig = plt.figure()
    for i in range(1,n_est):
        list_no_estimators.append(i)
        bg_clf = BaggingClassifier(base_estimator=dt_clf,
                                   n_estimators=i,
                                   random_state=9,
                                   bootstrap=True,
                                  max_samples=0.67,
                                  max_features=0.67)
        bg_clf.fit(X_train,y_train)
        y_pred_bg_test = bg_clf.predict(X_test)
        acc_score = accuracy_score(y_test,y_pred_bg_test)
        acc_test.append(acc_score)
        
        y_pred_bg_train = bg_clf.predict(X_train)
        acc_score = accuracy_score(y_train,y_pred_bg_train)
        acc_train.append(acc_score)
    plt.plot(list_no_estimators,acc_train)
    plt.plot(list_no_estimators,acc_test)
    plt.legend(['Train Accuracy','Test Accuracy'])
    plt.show()
    return fig


