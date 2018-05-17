import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

dataframe = pd.read_csv('data/loan_prediction.csv')

X = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

scores1=[]
scores2=[]
def bagging(X_train, X_test, y_train, y_test,n_est):
    n_est=51
    estimators=range(1,n_est)
    decision_clf = DecisionTreeClassifier()
    
    for est in estimators:
        bagging_clf = BaggingClassifier(decision_clf, n_estimators=est, max_samples=0.67,max_features=0.67, 
                                    bootstrap=True, random_state=9)
        bagging_clf.fit(X_train, y_train)
        # test line
        y_pred_bagging1 = bagging_clf.predict(X_test)
        score_bc_dt1 = accuracy_score(y_test, y_pred_bagging1)
        scores1.append(score_bc_dt1)
        # train line
        y_pred_bagging2 = bagging_clf.predict(X_train)
        score_bc_dt2 = accuracy_score(y_train, y_pred_bagging2)
        scores2.append(score_bc_dt2)
    
    plt.figure(figsize=(10, 6))
    plt.title('Bagging Info')
    plt.xlabel('Estimators')
    plt.ylabel('Scores')
    plt.plot(estimators,scores1,'g',label='test line', linewidth=3)
    plt.plot(estimators,scores2,'c',label='train line', linewidth=3)
    plt.legend()
    plt.show()


