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

decision_clf = DecisionTreeClassifier()

score_bc_lr = []
n_est = 0
def bagging(X_train, X_test, y_train, y_test,n_est):
    for i in range(1, 51):
        bagging_clf1 = BaggingClassifier(decision_clf, n_estimators=i, max_samples=0.67,
                                        bootstrap=True, random_state=9,max_features=0.67)
        bagging_clf1.fit(X_train, y_train)
        y_pred_bagging = bagging_clf1.predict(X_test)
        score_bc_lr.append(accuracy_score(y_test, y_pred_bagging))

    #print score_bc_lr
    i = list(range(1,51))
    #print i
    plt.plot(i,score_bc_lr)
    return plt.show()

#print bagging(X_train, X_test, y_train, y_test,n_est)
