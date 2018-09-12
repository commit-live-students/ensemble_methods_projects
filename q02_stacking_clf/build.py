# %load q02_stacking_clf/build.py
# Default imports
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Loading data
dataframe = pd.read_csv('data/loan_prediction.csv')
X = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

lr = LogisticRegression(random_state=9)
dt1 = DecisionTreeClassifier(random_state=9)
dt2 = DecisionTreeClassifier(random_state=9, max_depth=9)

bg_clf1 = BaggingClassifier(base_estimator=lr, n_estimators=100,
                           max_samples=100, bootstrap=True,
                           oob_score=True)

bg_clf2 = BaggingClassifier(base_estimator=dt1, n_estimators=100,
                           max_samples=100, bootstrap=True,
                           oob_score=True)

bg_clf3 = BaggingClassifier(base_estimator=dt2, n_estimators=100,
                           max_samples=100, bootstrap=True,
                           oob_score=True)

models = [bg_clf1,bg_clf2,bg_clf3]
# Write your code here
def stacking_clf(models, X_train, y_train, X_test, y_test):
    predictions = []
    for i in models:
        i.fit(X_train, y_train)
        pred = np.array(i.predict_proba(X_train))
        predictions.append(pred)
    X_bag_train = np.concatenate((predictions[0], predictions[1], predictions[2]), axis=1)
    predictions_test = []
    for j in models:
        pred = np.array(j.predict_proba(X_test))
        predictions_test.append(pred)
    X_bag_test = np.concatenate((predictions_test[0], predictions_test[1], predictions_test[2]), axis=1)
    predictions_bag_final = bg_clf1.fit(X_bag_train, y_train).predict(X_bag_test)
    return accuracy_score(y_test, predictions_bag_final)
stacking_clf(models, X_train, y_train, X_test, y_test)


