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
    bagging_clf = BaggingClassifier(DecisionTreeClassifier(random_state=9), n_estimators=n_est, max_samples=100,
                                 bootstrap=True, random_state=9, oob_score=True)

    decision_clf = DecisionTreeClassifier()

    # Fitting single decision tree
    decision_clf.fit(X_train, y_train)
    y_pred = decision_clf.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred)
    accuracy_train= accuracy_score(X_train, y_train)
    fig = plt.figure(figsize=(10, 7))
    plt.plot(n_est, accuracy_test, label="Train set")
    plt.plot(n_est, accuracy_train, label="Test Set")

    plt.xlabel('n_estimators')
    plt.ylabel('Accuracy')
    plt.show()
