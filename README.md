# Credit-Card-Fraud-Detection
Data Source: https://www.kaggle.com/mlg-ulb/creditcardfraud

Context
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

Content
The datasets contains transactions made by credit cards in September 2013 by european cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.

The dataset has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection.
More details on current and past projects on related topics are available on https://www.researchgate.net/project/Fraud-detection-5 and the page of the DefeatFraud project


1. Problem and Goals
Our goal is to implement 3 different machine learning models and one tensorflow deep learning model in order to classify and compare the results, to the highest possible degree of accuracy. After initial data exploration, we knew we would implement a logistic regression model, DecisionTree and Randomforest classifier model, and a tensorflow keras model. Some challenges we observed from the start were the huge imbalance in the dataset: frauds only account for 0.172% of fraud transactions. In this case, it is much worse to have false negatives than false positives in our predictions because false negatives mean that someone gets away with credit card fraud. False positives, on the other hand, merely cause a complication and possible hassle when a cardholder mmust verify that they did, in fact, complete said transaction (and not a thief).

2. Data Processing
As we know before, features V1-V28 have been transformed by PCA and scaled already. Whereas feature "Time" and "Amount" have not. And considering that we will analyze these two features with other V1-V28, they should better be scaled before we train our model using various algorithms. Here is why and how. Which scaling mehtod should we use? The Standard Scaler is not recommended as "Time" and "Amount" features are not normally distributed. The Min-Max Scaler is also not recommende as there are noticeable outliers in feature "Amount". The Robust Scaler are robust to outliers: (xi–Q1(x))/( Q3(x)–Q1(x)) (Q1 and Q3 represent 25% and 75% quartiles). So we choose Robust Scaler to scale these two features.

3. Data Modeling
The 4 models we used were a fully connected neural network, logistic regression, DecisionTree and RandomForest classifier.  Logistic regression outperformed both the DecisonTree and RandomForest. We believe that it is because of how the decision boundary changed with the class weights features. RandomForest next, and DecisionTree performed the poorest. 






