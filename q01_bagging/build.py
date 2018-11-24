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
    i=1
    train = dict()
    test = dict()
    
    while (i<=50):        
        bagging_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=i, max_samples=0.67, 
                                        bootstrap=True, random_state=9,max_features=0.67)
        model = bagging_clf.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        score_train = accuracy_score(y_train, y_pred_train)
        
        y_pred_test = model.predict(X_test)
        score_test = accuracy_score(y_test, y_pred_test)
        
        train[i]=score_train
        test[i]=score_test
        i= i + 1
    
    plt.plot(np.arange(1,51),train.values())
    plt.plot(np.arange(1,51),test.values())
    plt.show()

# bagging(X_train, X_test, y_train, y_test,n_est)


