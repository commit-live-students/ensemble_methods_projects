# Implement a Stacking classifier in a python
***
* We have learned what is Stacking, so now try to build Stacking Classifier.
* For this you need to recap your numpy skills, we have loaded the necessary data and packages for you.

##  Write a Function `stacking_clf` that:
* Will take three models, one BaggingClassifier with logistic regression and other BaggingClassifier's with two decision tree and build stackingclassifier using  **stacking_clf** function with meta_classifier as logistic regression.
* Use random state 9 for each model and max_depth = 9 for second decision tree.
* Use n_estimators, max_samples to 100 and bootstrap=True, oob_score set to True in BaggingClassifier.
* Their will be two stage, First stage will train on the training set and will convert train set to (429,6) numpy array,do the same with the test set and convert to (185,6) numpy array.
* Second stage will be fitting with these newly created numpy array with the meta classifier and perdict the output.


### Parameters:

| Parameter | dtype | argument type | default value | description |
| --- | --- | --- | --- | --- |
| X_train | DataFrame | compulsory | | Dataframe containing feature variables for training|
| X_test | DataFrame | compulsory | | Dataframe containing feature variables for testing|
| y_train | Series/DataFrame | compulsory | | Training dataset target Variable |
| y_test | Series/DataFrame | compulsory | | Testing dataset target Variable |
| model |  | compulsory | | Contains three model that have mention |

### Return parameter:

| Return | dtype | description |
| --- | --- | --- |
| Accuracy of the model | float | Accuracy of the model for test dataset |


Hint :
* You can use **accuracy_score**  to check the scores
* Function to use np.concatenate, for loop
