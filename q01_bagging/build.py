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
def bagging_results (X_train, X_test, y_train, y_test,n_est):
    model= BaggingClassifier(base_estimator=DecisionTreeClassifier(),random_state=9 ,n_estimators=  n_est,bootstrap= True,max_samples =0.67,max_features= 0.67)
 
 
    model.fit(X_train, y_train)
    y_pred= model.predict(X_test)
    y_pred_train= model.predict(X_train)    
    return [n_est,accuracy_score(y_test, y_pred),accuracy_score(y_train, y_pred_train)]
 

def bagging (X_train, X_test, y_train, y_test,n_est):
    results = map(lambda n_est :  bagging_results(X_train, X_test,y_train,y_test, n_est), range(1,50))
    results = pd.DataFrame(results)
    plt.plot(results[:][0],results[:][1])
    plt.plot(results[:][0],results[:][2])
    plt.show()




