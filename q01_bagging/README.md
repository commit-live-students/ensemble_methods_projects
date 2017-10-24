# Let's start Ensembling Methods with Bagging

- In this assignment you will learn to build a BaggingClassifier using DecisionTreeClassifier.

## Write a function bagging that:

- Takes input the features and target with default parameters.
- Then fits a bagging classifier with DecisionTreeClassifier with multiple n-estimators values such as in between 1-30.
- Plot a graph with n-estimators and the accuracy of model.
- Use random_state 9 and bootstrap=True

### Parameters:

| Parameter | dtype | argument type | default value | description |
| --- | --- | --- | --- | --- |
| X_train | DataFrame | compulsory | | Dataframe containing feature variables for training|
| X_test | DataFrame | compulsory | | Dataframe containing feature variables for testing|
| y_train | Series/DataFrame | compulsory | | Training dataset target Variable |
| y_test | Series/DataFrame | compulsory | | Testing dataset target Variable |
| n_est | int | compulsory | | n_estimators |

### Returns:
None
