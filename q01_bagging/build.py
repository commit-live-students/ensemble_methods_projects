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
    classifier = DecisionTreeClassifier(random_state=9)
    acc_test=[]
    acc_train=[]
    for i in range(1,50):
        bagClasifier= BaggingClassifier(base_estimator=classifier,n_estimators=i,max_samples=0.67,bootstrap=True,max_features=0.67,random_state=9)
        bagClasifier.fit(X_train,y_train)
        y_pred_train=bagClasifier.predict(X_train)
        y_pred_test=bagClasifier.predict(X_test)
        acc_score_test=accuracy_score(y_test, y_pred_test)
        acc_score_train=accuracy_score(y_train, y_pred_train)
        acc_test.append(acc_score_test)
        acc_train.append(acc_score_train)
        #print(acc_score_test)
    #print('printing')
    xaxis=range(1,50)
    #print(xaxis)
    #print(acc_test)
    plt.plot(xaxis,acc_test)
    plt.plot(xaxis,acc_train)
    plt.show()


#bagging(X_train, X_test, y_train, y_test, 50)
