# Default imports
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
# Data Loading
dataframe = pd.read_csv('data/loan_prediction.csv')


#Solution
# Write your code here
