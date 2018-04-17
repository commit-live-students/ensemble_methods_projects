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
    lst = []
    for i in range(1,n_est):
        bagging_clf2 = BaggingClassifier(DecisionTreeClassifier(), n_estimators=i, max_samples=0.67,
                                        bootstrap=True, random_state=9, max_features=0.67)

        bagging_clf2.fit(X_train, y_train)
        y_pred_test = bagging_clf2.predict(X_test)
        y_pred_train = bagging_clf2.predict(X_train)
        lst.append((i,accuracy_score(y_test, y_pred_test),accuracy_score(y_train,y_pred_train)))
    df = pd.DataFrame(lst)
    plt.xlabel('n_estimators')
    plt.ylabel('accuracy')
    plt.plot(df.iloc[:,0],df.iloc[:,2],c='b', label='Train set')
    plt.plot(df.iloc[:,0],df.iloc[:,1],c='g', label='Test set')
    plt.legend()
    plt.show()
    return
