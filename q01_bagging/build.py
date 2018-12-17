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
def bagging(X_train,X_test,y_train,y_test,n_est):
    dt_classifier = DecisionTreeClassifier(random_state=9)

    error_train = []
    error_test = []
    for i in range(1,51):
        bg_classifier = BaggingClassifier(n_estimators=i,base_estimator=dt_classifier,bootstrap=True,max_features=0.67,max_samples=0.67)

        bg_classifier.fit(X_train,y_train)

        predict_train = bg_classifier.predict(X_train)
        predict_test = bg_classifier.predict(X_test)

        error_train.append(accuracy_score(y_train,predict_train))
        error_test.append(accuracy_score(y_test,predict_test))

    plt.plot(range(1,51),error_train,label='Train')

    plt.plot(range(1,51),error_test,label='Test')
    plt.legend()
    plt.show()


