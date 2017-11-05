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
    n_est=50
    dtc = DecisionTreeClassifier()
    x=[]
    y=[]
    y_sample_train=[]
    for i in range(1,n_est):
        bagging_clf1 = BaggingClassifier(dtc, n_estimators=i, max_samples=0.67,
                                bootstrap=True, max_features=0.67,random_state=9)
        bagging_clf1.fit(X_train, y_train)
        y_pred_bagging = bagging_clf1.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_bagging)
        x.append(i)
        y.append(accuracy)
        y_train_pred_bagging = bagging_clf1.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred_bagging)
        y_sample_train.append(train_accuracy)
    plt.plot(x,y)
    plt.plot(x,y_sample_train)
    plt.show()
