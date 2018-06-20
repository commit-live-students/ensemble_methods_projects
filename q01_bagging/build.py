import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Data Loading
dataframe = pd.read_csv('data/loan_prediction.csv')

X = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)


# Write your code here
values = list(range(1,50))
def bagging(X_train, X_test, y_train, y_test,n_est):
    final=[]
    def bagging_subset(n_est):
        bagging_clf1 = BaggingClassifier(DecisionTreeClassifier(), n_estimators=n_est, max_samples=0.67,max_features=0.67,
                                bootstrap=True, random_state=9)
        bagging_clf1.fit(X_train, y_train)
        y_pred_bagtest = bagging_clf1.predict(X_test)
        score_test = accuracy_score(y_test, y_pred_bagtest)

        y_pred_bagtrain = bagging_clf1.predict(X_train)
        score_train = accuracy_score(y_train, y_pred_bagtrain)
        #print n_est1, score_test, score_train
        return (n_est, score_test, score_train)

    results = map(lambda n_est :  bagging_subset(n_est), values)
    results = pd.DataFrame(results)

    plt.plot(results[:][0],results[:][1])
    plt.plot(results[:][0],results[:][2])
    plt.show()
    return
