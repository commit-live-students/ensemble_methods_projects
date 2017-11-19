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
def bagging(X_train,X_test,y_train,y_test,n_est):
    accs1 = []
    accs2 = []
    for i in range (1,n_est+1):
        bagging_clf = BaggingClassifier(DecisionTreeClassifier(),n_estimators=i, bootstrap=True,max_samples=0.67,max_features=0.67,random_state=9)
        bagging_clf.fit(X_train,y_train)
        y_pred1 = bagging_clf.predict(X_test)
        acc1 = accuracy_score(y_test,y_pred1)
        y_pred2 = bagging_clf.predict(X_train)
        acc2 = accuracy_score(y_train,y_pred2)
        accs1.append(acc1)
        accs2.append(acc2)
    plt.plot(range(1,51),accs1)
    plt.plot(range(1,51),accs2)
    plt.show()
