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

def bagging(X_train, X_test, y_train, y_test, n_est):
    acc_score_train = []
    acc_score_test = []
    for i in range(1,n_est+1):
        bagging_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=i, max_samples=0.67,max_features = 0.67,
                                bootstrap=True, random_state=9)
        bagging_clf.fit(X_train, y_train)

        y_test_pred = bagging_clf.predict(X_test)
        y_train_pred = bagging_clf.predict(X_train)

        acc_score_train.append(accuracy_score(y_train, y_train_pred))
        acc_score_test.append(accuracy_score(y_test, y_test_pred))


    plt.plot(range(1,n_est+1),acc_score_train, label = 'Train Set')
    plt.plot(range(1,n_est+1),acc_score_test, label = 'Test Set')
    plt.legend()
    plt.xlabel('No. of estimators')
    plt.ylabel('Accuracy')

    plt.show()
# Write your code here
