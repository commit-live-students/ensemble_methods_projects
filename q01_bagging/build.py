# %load q01_bagging/build.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np

# Data Loading
dataframe = pd.read_csv('data/loan_prediction.csv')

X = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)


# Write your code here


def bagging(X_train, X_test, y_train, y_test,n_est):
    
    i=1
    dict1=dict()
    dict2=dict()
    
    while (i<=50):        
        # Fitting bagging classifier with Logisitc Regression
        bagging_clf2 = BaggingClassifier(DecisionTreeClassifier(), n_estimators=i, max_samples=0.67, 
                                        bootstrap=True, random_state=9,max_features=0.67)

        bagging_clf2.fit(X_train, y_train)
        y_pred_bagging_t = bagging_clf2.predict(X_train)
        score_bc_dt_t = accuracy_score(y_train, y_pred_bagging_t)
        y_pred_bagging = bagging_clf2.predict(X_test)
        score_bc_dt = accuracy_score(y_test, y_pred_bagging)
        dict1[i]=score_bc_dt_t
        dict2[i]=score_bc_dt
        i+=1
    
    plt.plot(np.arange(1,51),dict1.values())
    plt.plot(np.arange(1,51),dict2.values())
    plt.show()




