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
def bagging(X_train, X_test, y_train, y_test,n):
    test_score=[]
    train_score=[]
    n_est=range(1,n)
    for i in n_est:
        model=DecisionTreeClassifier(random_state=9)
        bag=BaggingClassifier(model, n_estimators=i,max_samples=0.67,n_jobs=-1,max_features=0.67,bootstrap=True)
        bag.fit(X_train,y_train)
        y_pred=bag.predict(X_train)
        y_pred_test=bag.predict(X_test)
        train_score.insert(i,accuracy_score(y_train,y_pred))
        test_score.insert(i,accuracy_score(y_test,y_pred_test))
# Write your code here
    plt.plot(n_est,train_score,'g',label='train-set')
    plt.plot(n_est,test_score,'b',label='test-set')
    plt.xlabel('n_estimators')
    plt.ylabel('accuracy_score')
    plt.legend('best')
    plt.show()
