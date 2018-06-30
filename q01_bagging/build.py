
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
def bagging(X_train, X_test, y_train, y_test,n_est):
    accuracy_scores=[]
    accuracy_scores_t=[]
    n_est_list=[]
    for n in list(range(1,n_est,1)):
        bagging_clf1 = BaggingClassifier(DecisionTreeClassifier(), n_estimators=n, max_samples=0.67,max_features=0.67, 
                                    bootstrap=True, random_state=9)
        bagging_clf1.fit(X_train, y_train)
        y_pred_bagging = bagging_clf1.predict(X_test)
        score_bc_lr = accuracy_score(y_test, y_pred_bagging)
        accuracy_scores.append(score_bc_lr)
        
        y_pred_bagging_t = bagging_clf1.predict(X_train)
        score_bc_lr_t = accuracy_score(y_train, y_pred_bagging_t)
        accuracy_scores_t.append(score_bc_lr_t)
        n_est_list.append(n)
        
    plt.plot(n_est_list,accuracy_scores_t,accuracy_scores)
    #plt.legend('Train Set','Test Set')
    plt.show()

#bagging(X_train, X_test, y_train, y_test,50)



