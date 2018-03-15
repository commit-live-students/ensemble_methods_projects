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

r1=[]
r2=[]
# Write your code here
def bagging(X_train,X_test,y_train,y_test,n_est):
    for est in range(1,n_est):
        bagging_clf2 = BaggingClassifier(DecisionTreeClassifier(), n_estimators=est,max_samples=0.67, max_features=0.67,
                                bootstrap=True, random_state=9)

        bagging_clf2.fit(X_train, y_train)
        y_pred_tr = bagging_clf2.predict(X_train)
        y_pred_ts = bagging_clf2.predict(X_test)

        r1.append(accuracy_score(y_train, y_pred_tr))
        r2.append(accuracy_score(y_test, y_pred_ts))

    plt.plot(range(1,n_est),r1)
    plt.plot(range(1,n_est),r2)
    plt.show()
