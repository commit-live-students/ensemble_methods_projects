# %load q02_stacking_clf/build.py
# Default imports
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from mlxtend.classifier import StackingClassifier
import pandas as pd
import numpy as np

# Loading data
dataframe = pd.read_csv('data/loan_prediction.csv')
X = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)
LR=LogisticRegression(random_state=9)
DT1=DecisionTreeClassifier(random_state=9)
DT2=DecisionTreeClassifier(max_depth=9,random_state=9)
bagging_clf1 = BaggingClassifier(LR, n_estimators=100, max_samples=100, 
                                bootstrap=True, random_state=9,oob_score=True)
bagging_clf2 = BaggingClassifier(DT1, n_estimators=100, max_samples=100, 
                                bootstrap=True,oob_score=True)

bagging_clf3 = BaggingClassifier(DT2, n_estimators=100, max_samples=100, 
                                bootstrap=True, random_state=9,oob_score=True)



def ModelUse(ModelToUse):
    ModelToUse.fit(X_train, y_train)
    y_pred_decision=ModelToUse.predict(X_test)    
    score=accuracy_score(y_test.reshape(-1,1),y_pred_decision)
    y_pred_decision=y_pred_decision.reshape(185,1)
    NewXtest=np.concatenate((X_test, y_pred_decision), axis=1)
    y_pred_decision1=ModelToUse.predict(X_train)    
    y_pred_decision1=y_pred_decision1.reshape(429,1)
    NewXtrain=np.concatenate((X_train, y_pred_decision1), axis=1)
    return(NewXtest,NewXtrain)

model=[bagging_clf1,bagging_clf2,bagging_clf3]

def stacking_clf(model,Xtrain,y_train,Xtest,y_test):
    stacking_clf = StackingClassifier(classifiers = model,
                                 meta_classifier = LR)
    stacking_clf.fit(NewXtrain, y_train)
    y_pred2 = stacking_clf.predict(NewXtest)
    accuracy = accuracy_score(y_test, y_pred2)
    return(accuracy)

NewXtest,NewXtrain=ModelUse(bagging_clf1)
stacking_clf(model,NewXtrain,y_train,NewXtest,y_test)
NewXtest,NewXtrain=ModelUse(bagging_clf2)
stacking_clf(model,NewXtrain,y_train,NewXtest,y_test)
NewXtest,NewXtrain=ModelUse(bagging_clf3)
stacking_clf(model,NewXtrain,y_train,NewXtest,y_test)

