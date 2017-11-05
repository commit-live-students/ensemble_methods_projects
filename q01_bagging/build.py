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
# clive open ensemble_methods_projects:q01_bagging
def bagging(X_train, X_test, y_train, y_test,n_est):
    values=[]
    values_tr=[]
    for val in n_est:
        model=BaggingClassifier(base_estimator=None, n_estimators=val, max_samples=0.67, max_features=0.67, bootstrap=True, random_state=9)
        model.fit(X_train,y_train)
        ypred=model.predict(X_test)
        xpred=model.predict(X_train)
        ac=accuracy_score(y_test,ypred)
        actr=accuracy_score(y_train,xpred)
        values.append(ac)
        values_tr.append(actr)
    plt.plot(n_est,values)

    plt.plot(n_est,values_tr)
    plt.show()
    
