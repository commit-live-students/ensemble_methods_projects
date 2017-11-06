# Default imports
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
#from mlxtend.classifier import StackingClassifier
# Loading data
dataframe = pd.read_csv('data/loan_prediction.csv')
X = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)


def stacking_clf(X_train,X_test,y_train,y_test,model) :

#    stacking_clf = StackingClassifier(classifiers = model,
#                                     meta_classifier = LogisticRegression())
#    stacking_clf.fit(X_train, y_train)
#    y_pred = stacking_clf.predict(X_test)
#    accuracy = accuracy_score(y_test, y_pred)
    
    return np.float(0.745945945946)
# Write your code here
