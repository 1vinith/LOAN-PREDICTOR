# LOAN-PREDICTOR

This is a basic machine learning classification project intended to learn and implement different machine learning algorithms using sklearn along with regularizations and validation techniques to improve scores and reduce over fitting. More additions will be done to it in the coming time. This repository is intended to be used as a collective reference for implementation of various machine learning algorithms and techniques. The algorithms included are -

1.Logistic Regression classifier
2.SGD classifier
3.SVM - Linear/RBF
4.KNN
5.Decision Tree Classifier
6.Random Forest classifier
7.Ada-Boost
8.XgBoost
9.Naive Bayes
10.Gaussian Process classifier
11.Voting classifier

Here is a theoritical information about the function and variable used in deriving this whole program

<h3>Data Loading and Preprocessing:</h3>

The script starts by importing necessary libraries such as NumPy, Pandas, and scikit-learn for machine learning tasks.
It loads training and test datasets ('train.csv' and 'test.csv') using Pandas DataFrames.
Another function named encodingCategColumns is defined for one-hot encoding categorical columns and label encoding the target column.
<h2>Handling Missing Values:</h2>
The handleNullValues function is called to fill missing values in both the training and test datasets.

Encoding Categorical Columns:

The encodingCategColumns function is used to one-hot encode categorical columns and label encode the target column in the training dataset.
The same process is applied to the test dataset.
<h3>Scaling Data:</h3>
The script uses Min-Max scaling to scale the features in both the training and test datasets.
<h3>Train-Test Split:</h3>
The dataset is split into training and testing sets using the train_test_split function.

<h3>Building Classification Models:</h3>

Several classification algorithms are implemented and tuned using either GridSearchCV or RandomizedSearchCV for hyperparameter optimization.
The implemented classifiers include Logistic Regression, Stochastic Gradient Descent (SGD) Classifier, Support Vector Machine (SVM) with both linear and radial basis function (RBF) kernels, k-Nearest Neighbors (KNN), Decision Tree Classifier, Random Forest Classifier, AdaBoost Classifier, XGBoost, Naive Bayes, Gaussian Process 
<h3>Ensemble Techniques:</h3>
Ensemble techniques like AdaBoost, XGBoost, and a Voting Classifier are implemented to combine the predictions of multiple base classifiers.
Evaluation Metrics:
The script calculates and prints various evaluation metrics such as accuracy, ROC AUC score, and classification reports for individual classifiers and ensemble models.
<h3></h3>Voting Classifier Across Folds:</h3>
The script demonstrates the use of a Voting Classifier across folds, calculating accuracy scores, classification reports, and confusion matrices for each fold.
