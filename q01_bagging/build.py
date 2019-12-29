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


# Write your code here
def bagging(X_train, X_test, y_train, y_test, n_est):
    dtree = DecisionTreeClassifier(random_state=9)
    lst_train_score = list()
    lst_test_score = list()
    estimator_arr = range(1,n_est,2)
    for est_val in estimator_arr:
        bagging_clf = BaggingClassifier(base_estimator=dtree, n_estimators=est_val,\
                                    max_samples=0.67, max_features=0.67, bootstrap=True,\
                                    random_state=9)
        bagging_clf.fit(X_train, y_train)
        y_pred_test = bagging_clf.predict(X_test)
        y_pred_train = bagging_clf.predict(X_train)
        accuracy_score_train = accuracy_score(y_true=y_train, y_pred=y_pred_train)
        accuracy_score_test = accuracy_score(y_true=y_test, y_pred=y_pred_test)
        lst_train_score.append(accuracy_score_train)
        lst_test_score.append(accuracy_score_test)
    plt.plot(estimator_arr, lst_train_score, color='b',label='Train Accuracy')
    plt.plot(estimator_arr, lst_test_score, color='g',label='Test Accuracy')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy Score')
    plt.legend()
    plt.show() 
