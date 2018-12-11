# %load q01_bagging/build.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
plt.switch_backend('agg')

dataframe = pd.read_csv('data/loan_prediction.csv')
X = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

def bagging(X_train, X_test, y_train, y_test, n_est):
    accuracy_test, accuracy_train  = [], []
    for i in range(1,n_est+1):
        bagging_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=i,
                                   random_state=9, bootstrap=True, 
                                        max_samples=0.67, max_features=0.67)
        bagging_clf.fit(X_train, y_train)
        y_pred_test = bagging_clf.predict(X_test)
        y_pred_train = bagging_clf.predict(X_train)
        accuracy_test.append(accuracy_score(y_test,y_pred_test))
        accuracy_train.append(accuracy_score(y_train,y_pred_train))
    plt.plot(range(1,51), accuracy_test, label='test', color='blue')
    plt.plot(range(1,51), accuracy_train, label='train', color='red')
    plt.xlabel('n-estimators')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show();



