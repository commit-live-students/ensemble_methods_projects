# %load q01_bagging/build.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
plt.switch_backend('agg')
# Data Loading
dataframe = pd.read_csv('data/loan_prediction.csv')

X = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)
nest=[]
for i in range(1,51):
    nest.append(i)
ac1 = []
ac2 = []
n_est=None
def bagging(X_train,X_test,y_train,y_test,n_est):
    for i in nest:
        dtc = DecisionTreeClassifier()
        bc = BaggingClassifier(base_estimator=dtc,n_estimators=i,max_samples=0.67,max_features=0.67,random_state=9)
        model = bc.fit(X_train,y_train)
        y_pred = model.predict(X_train)
        a = accuracy_score(y_train,y_pred)
        ac1.append(a)

        y_pred2 = model.predict(X_test)
        b = accuracy_score(y_test,y_pred2)
        ac2.append(b)
    plt.plot(nest,ac1)
    plt.plot(nest,ac2)
        
c = bagging(X_train,X_test,y_train,y_test,n_est)



