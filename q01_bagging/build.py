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
def bagging(X_train, X_test, y_train, y_test, n_est ):
    
    test_score = []
    train_score = []
    n_esti = range(1,n_est)
    for  i in n_esti:
        bagging_clf2 = BaggingClassifier(DecisionTreeClassifier(), n_estimators=i, max_samples=0.67, 
                                        bootstrap=True, random_state=9 ,max_features = 0.67,)

        bagging_clf2.fit(X_train, y_train)
        y_pred_bagging = bagging_clf2.predict(X_test)
        test_score.append( accuracy_score(y_test, y_pred_bagging))

        y_pred_bagging1 = bagging_clf2.predict(X_train)
        train_score.append( accuracy_score(y_train, y_pred_bagging1))
        
    plt.plot(n_esti, test_score , label = 'Test_set')
    plt.plot(n_esti,train_score, label = 'Train_set')
    plt.xlabel('n_estimators')

    plt.ylabel('accuracy')
    plt.legend(loc=1)
    plt.show()



bagging(X_train, X_test, y_train, y_test, 50 )


