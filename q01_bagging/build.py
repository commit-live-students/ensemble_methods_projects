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
    model = DecisionTreeClassifier()
    train_score = []
    test_score = []
    for x in range(1,n_est+1):
        baggingclf = BaggingClassifier(model, n_estimators=x, max_samples=0.67, max_features=0.67, bootstrap=True, random_state=9)
        baggingclf.fit(X_train, y_train)
        y_train_pred = baggingclf.predict(X_train)
        train_score.append(accuracy_score(y_train, y_train_pred))
        y_test_pred = baggingclf.predict(X_test)
        test_score.append(accuracy_score(y_test, y_test_pred))
    plt.plot(range(1,n_est+1), train_score, "r", label="Training scores")
    plt.plot(range(1,n_est+1), test_score, "b", label="Testing score")
