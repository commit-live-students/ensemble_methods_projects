# %load q01_bagging/build.py
import pandas as pd
import numpy as np
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
estimator = np.arange(1, 51)

# Write your code here
def bagging(X_train, X_test, y_train, y_test, estimator):
    train_accuracy = []
    test_accuracy = []
    
    for est in estimator:
        bc = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=est, 
                               max_samples=0.67, bootstrap=True, random_state=9,
                               max_features=0.67)
        model = bc.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        train_accuracy.append(accuracy_score(y_train, y_train_pred))

        y_test_pred = model.predict(X_test)
        test_accuracy.append(accuracy_score(y_test, y_test_pred))
    
    plt.plot(estimator, train_accuracy)
    plt.plot(estimator, test_accuracy)
    plt.show()

bagging(X_train, X_test, y_train, y_test, estimator)


