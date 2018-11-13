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
def bagging(X_train, X_test, y_train, y_test,n_est):
    clf = BaggingClassifier(DecisionTreeClassifier(),n_estimators=50,max_sample=0.67,bootstrap=True,random_state=9)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test,y_pred)
    plt.plot(n_est,score)
    plt.show()


