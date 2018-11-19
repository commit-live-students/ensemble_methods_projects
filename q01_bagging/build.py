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
# def bagging(X_train,X_test,y_train,y_test,n_est):
    
#     dt = DecisionTreeClassifier()
#     score_test = []
#     score_train = []
#     for x in range(1,50):
    
#         bag = BaggingClassifier(base_estimator = dt, n_estimators = x,random_state=9, bootstrap= True, max_features=0.67,max_samples=0.67)

#         bag.fit(X_train,y_train)
#         score_test.append(bag.score(X_test,y_test))
#         score_train.append(bag.score(X_train,y_train))

    
#         plt.plot(score_test)
#         plt.plot(score_train)
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
    acc_train, acc_test = [], []
    ranges = range(1, n_est, 2)
    for n_est in ranges:
        clf_bagging_20 = BaggingClassifier(DecisionTreeClassifier(), n_estimators=n_est, max_samples=0.67,
                                           max_features=0.67, bootstrap=True, random_state=9)
        clf_bagging_20.fit(X_train, y_train)
        X_pred_bagging = clf_bagging_20.predict(X_train)
        y_pred_bagging = clf_bagging_20.predict(X_test)
        acc_score_train = accuracy_score(y_train, X_pred_bagging)
        acc_score_test = accuracy_score(y_test, y_pred_bagging)
        acc_train.append(acc_score_train)
        acc_test.append(acc_score_test)
    plt.figure(figsize=(10, 6))
    plt.plot(ranges, acc_train, c='b', label='Train set')
    plt.plot(ranges, acc_test, c='g', label='Test set')
    plt.legend(loc='upper right')
    plt.xlabel('n_estimators')
    plt.ylabel('Accuracy')
    plt.show()






