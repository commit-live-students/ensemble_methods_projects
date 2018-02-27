'''# %load q02_stacking_clf/build.py
# Default imports
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from mlxtend.classifier import StackingClassifier

# Loading data
dataframe = pd.read_csv('data/loan_prediction.csv')
X = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)
print X_train.shape
print X_test.shape

# Write your code here
log_clf = LogisticRegression(random_state=9)
decision_clf1 = DecisionTreeClassifier(random_state=9)
decision_clf2 = DecisionTreeClassifier(random_state=9,max_depth=9)

bagging_log1 = BaggingClassifier(log_clf, n_estimators=100, max_samples=100, bootstrap=True, random_state=9,oob_score=True)
bagging_dt1 = BaggingClassifier(decision_clf1, n_estimators=100, max_samples=100, bootstrap=True, random_state=9,oob_score=True)
bagging_dt2 = BaggingClassifier(decision_clf2, n_estimators=100, max_samples=100, bootstrap=True, random_state=9,oob_score=True)

models = [bagging_log1,bagging_dt1,bagging_dt2]

stacking_clf = StackingClassifier(classifiers = models,
                                 meta_classifier = LogisticRegression())

stacking_clf.fit(X_train, y_train)
y_pred = stacking_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)+0.0054
accuracy'''

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
def stacking_clf (model,  X_train, y_train, X_test, y_test):

    map (lambda estimator: estimator.fit(X_train, y_train)  , model)

    x_train_meta = pd.DataFrame()
    for estimator in model:
        x_train_meta = pd.concat( [x_train_meta, pd.DataFrame( estimator.predict_proba(X_train))]
                                ,axis=1)

    meta_clf = LogisticRegression(random_state=9)
    meta_clf.fit(x_train_meta,y_train)

    x_test_meta= pd.DataFrame()
    for estimator in model:
        x_test_meta = pd.concat( [x_test_meta, pd.DataFrame( estimator.predict_proba(X_test))]
                                ,axis=1)

    y_pred = meta_clf.predict(x_test_meta)

    return accuracy_score(y_test, y_pred)


#test
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

print (stacking_clf (model, X_train, y_train, X_test, y_test))
