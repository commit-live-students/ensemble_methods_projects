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
def bagging(X_train, X_test, y_train, y_test, n_est):
    model = DecisionTreeClassifier(random_state=9)
    bag = BaggingClassifier(base_estimator=model, n_estimators=n_est)
    bag.fit(X_train, y_train)
    y_pred = bag.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

acc = []
x = []
for i in range(1,51):
    x.append(i)
    acc.append(bagging(X_train, X_test, y_train, y_test, i))

plt.plot(x=x, y=acc)
plt.show()
