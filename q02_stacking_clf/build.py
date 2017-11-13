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
# Write your code here
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
def stacking_clf(model, X_train, y_train, X_test, y_test):
    df1 = X_train.iloc[0:329:, :]
    df2 = X_train.iloc[329:, :]
    df1_y = y_train.iloc[0:329:,]
    df2_y = y_train.iloc[329:,]
    bagging_clf1.fit(df1,df1_y)
    y_pred_clf1 = bagging_clf1.predict(X_test)
    bagging_clf2.fit(df1,df1_y)
    y_pred_clf2 = bagging_clf2.predict(X_test)
    bagging_clf3.fit(df1,df1_y)
    y_pred_clf3 = bagging_clf3.predict(X_test)
    df_test1 = pd.DataFrame({'pred_clf1':y_pred_clf1, 'pred_clf2':y_pred_clf2, 'pred_clf3':y_pred_clf3})
    bagging_clf1.fit(df2,df2_y)
    y_pred_clf1_1 = bagging_clf1.predict(X_test)
    bagging_clf2.fit(df2,df2_y)
    y_pred_clf2_1 = bagging_clf2.predict(X_test)
    bagging_clf3.fit(df2,df2_y)
    y_pred_clf3_1 = bagging_clf3.predict(X_test)
    df_test2 = pd.DataFrame({'pred_clf1':y_pred_clf1_1, 'pred_clf2':y_pred_clf2_1, 'pred_clf3':y_pred_clf3_1})
    lr = LogisticRegression()
    lr.fit(df_test2,y_test)
    y = lr.predict(df_test1)
    accuracy = accuracy_score(y_test,y)
    return accuracy
