import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

n_est = range(1, 50)
# Data Loading
dataframe = pd.read_csv('data/loan_prediction.csv')

X = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

n_est = range(1, 50)
# Write your code here
def bagging_1(X_train, X_test, y_train, y_test, n_est):
    model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=9, n_estimators= n_est, bootstrap=True, max_samples=0.67, max_features=0.67)
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
    y_pred_test = model.predict(X_train)
    return (n_est, accuracy_score(y_test, y_pred), accuracy_score(y_train, y_pred_test))

def bagging(X_train, X_test, y_train, y_test, n_est):
    result = list(map(lambda n_est: bagging_1(X_train, X_test, y_train, y_test, n_est), n_est))
    result = pd.DataFrame(result)
    plt.plot(result[:][0], result[:][1])
    plt.plot(result[:][0], result[:][2])
    plt.show()

#bagging(X_train, X_test, y_train, y_test, n_est)
