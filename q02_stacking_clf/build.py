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

# Write your code here
def stacking_clf(model, X_train, y_train, X_test, y_test):
    X_train_meta = pd.DataFrame()
    X_test_meta = pd.DataFrame()
    for model_ in model:
        # fit the models passed to method, using X_train and y_train
        model_.fit(X_train,y_train)
        # create train dataframe for Meta Classifier using models passed to the method
        # predict the probabilties on train
        df_meta_train = pd.DataFrame(model_.predict_proba(X_train))
        X_train_meta = pd.concat([X_train_meta, df_meta_train],axis=1)

        # create test dataframe for Meta Classifier using models passed to the method
        # predict the probabilties on test
        df_meta_test = pd.DataFrame(model_.predict_proba(X_test))
        X_test_meta = pd.concat([X_test_meta, df_meta_test],axis=1)

    # fit metaclassifier using Logistic
    meta_logcf = LogisticRegression(random_state=9)
    meta_logcf.fit(X_train_meta,y_train)
    # Predict using metaclassifier using Logistic
    y_pred_meta_test = meta_logcf.predict(X_test_meta)
    acc_score = accuracy_score(y_true=y_test, y_pred=y_pred_meta_test)
    return acc_score
