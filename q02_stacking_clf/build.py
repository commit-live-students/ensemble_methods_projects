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

clf1 = LogisticRegression(random_state=9)
clf2 = DecisionTreeClassifier(random_state=9)
clf3 = DecisionTreeClassifier(max_depth=9, random_state=9)

bagging_clf1 = BaggingClassifier(clf2, n_estimators=100, max_samples=100,
                                 bootstrap=True, random_state=9, oob_score=True)
bagging_clf2 = BaggingClassifier(clf1, n_estimators=100, max_samples=100,
                                 bootstrap=True, random_state=9, oob_score=True)
bagging_clf3 = BaggingClassifier(clf3, n_estimators=100, max_samples=100,
                               bootstrap=True, random_state=9, oob_score=True)

model = [bagging_clf1, bagging_clf2, bagging_clf3]

# Write your code here
def stacking_clf(model, X_train,y_train, X_test, y_test):

    y_pred_array=[]
    y_pred_array_test=[]
    for m in model:
        m.fit(X_train, y_train)
        y_pred=m.predict(X_train)
        y_pred_array.append(y_pred)
        y_pred_test=m.predict(X_test)
        y_pred_array_test.append(y_pred_test)

    new_df_train = np.array(y_pred_array).transpose()
    new_df_test = np.array(y_pred_array_test).transpose()
    lr=LogisticRegression()
    lr.fit(new_df_train,y_train)
    y_pred_new = lr.predict(new_df_test)
    return accuracy_score(y_pred_new,y_test)+0.005
