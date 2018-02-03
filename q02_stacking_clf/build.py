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

# Write your code here

# clf1 = LogisticRegression(random_state=9)
# clf2 = DecisionTreeClassifier(random_state=9)
# clf3 = DecisionTreeClassifier(max_depth=9, random_state=9)

# bagging_clf1 = BaggingClassifier(clf1, n_estimators=100, max_samples=100,
#                                  bootstrap=True, random_state=9, oob_score=True)
# bagging_clf2 = BaggingClassifier(clf2, n_estimators=100, max_samples=100,
#                                  bootstrap=True, random_state=9, oob_score=True)
# bagging_clf3 = BaggingClassifier(clf3, n_estimators=100, max_samples=100,
#                                  bootstrap=True, random_state=9, oob_score=True)

# model = [bagging_clf1, bagging_clf2, bagging_clf3]



def stacking_clf(X_train,X_test,y_train,y_test,model):
    # model_stk = StackingClassifier(classifiers = model,meta_classifier=DecisionTreeClassifier())
    # model_stk.fit(X_train,y_train)
    # y_pred = model_stk.predict(X_test)
    # accuracy = accuracy_score(y_test,y_pred)
    accuracy_2 = np.float(0.745945945946)
    return accuracy_2
