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
def bagging(X_train, X_test, y_train, y_test,n_est = 10):
    decision_clf = DecisionTreeClassifier()
    
    # Fitting single decision tree
    decision_clf.fit(X_train, y_train)
    y_pred_decision = decision_clf.predict(X_test)
    score_dt1 = accuracy_score(y_test, y_pred_decision)
    
    
    # Fitting bagging classifier with DecisionTreeClassifier
    bagging_clf1 = BaggingClassifier(decision_clf, n_est, max_samples=0.67,max_features=0.67, 
                                bootstrap=True, random_state=9)

    bagging_clf1.fit(X_train, y_train)
    y_pred_bagging = bagging_clf1.predict(X_test)
    score_bc_dt = accuracy_score(y_test, y_pred_bagging)
    
    return plt.plot(n_est,score_bc_dt)
    



print(bagging(X_train, X_test, y_train, y_test,n_est = 10))
plt.show()

