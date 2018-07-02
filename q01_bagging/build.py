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

accuracy_train_list =[]
accuracy_test_list =[]
estimator_list = []

# Write your code here
def bagging(X_train,X_test, y_train, y_test,n_est):
    for i in range(1,50):
        bag_clf = BaggingClassifier(DecisionTreeClassifier(),
                                  n_estimators=i,
                                  max_samples = 0.67,
                                  max_features = 0.67,
                                  bootstrap = True,
                                  random_state=9)
        bag_clf.fit(X_train,y_train)
        y_test_pred = bag_clf.predict(X_test)
        bag_clf.fit(X_test,y_test)
        y_train_pred = bag_clf.predict(X_train)
        accuracy_test_list.append(accuracy_score(y_test,y_test_pred))
        accuracy_train_list.append(accuracy_score(y_train,y_train_pred))
        estimator_list.append(i)

    accuracy_test_list.pop(0)    
    accuracy_train_list.pop(0)
    estimator_list.pop(0)    
    plt.plot(estimator_list,accuracy_train_list)
    plt.plot(estimator_list,accuracy_test_list)
    plt.xlabel('n_estimator')
    plt.ylabel('accuracy')
    plt.show()

