# Default imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV

# Loading data
dataframe = pd.read_csv('data/loan_prediction.csv')


#Solution
# Write your code here
