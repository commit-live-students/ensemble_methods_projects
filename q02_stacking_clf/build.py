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
