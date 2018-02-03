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
def bagging(X_train,X_test,y_train,y_test,n_est):

    model = DecisionTreeClassifier(random_state=9)

    train_acc = []
    test_acc = []

    for i in range(n_est):

            model2 = BaggingClassifier(model,i+1,max_samples=0.67,max_features=0.67,random_state=9,bootstrap=True)
            model2.fit(X_train,y_train)
            y_pred_train = model2.predict(X_train)
            y_pred_test = model2.predict(X_test)
            train_acc.append(accuracy_score(y_train,y_pred_train))
            test_acc.append(accuracy_score(y_test,y_pred_test))
    plt.plot(range(n_est),train_acc)
    plt.plot(range(n_est),test_acc)
    plt.show()
    
