# %load q01_bagging/build.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
plt.switch_backend('agg')

# Data Loading
dataframe = pd.read_csv('data/loan_prediction.csv')

X = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)


# Write your code here
def bagging(X_train,X_test,y_train,y_test,n_est):
    scores_train=[]
    scores_test=[]
    for est in range(2,n_est):
        bag_clf = BaggingClassifier(DecisionTreeClassifier(random_state=9),n_estimators=est,random_state=9,bootstrap=True)
        bag_clf.fit(X_train,y_train)
        score_train = bag_clf.score(X_train,y_train)
        score_test = bag_clf.score(X_test,y_test)
        scores_test.append(score_test)
        scores_train.append(score_train)
    plt.plot(range(2,50),scores_train)
    plt.plot(range(2,50),scores_test)
bagging(X_train,X_test,y_train,y_test,50)


