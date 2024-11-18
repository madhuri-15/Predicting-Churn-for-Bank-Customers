# Predicting Churn for Bank Customers

This project focuses on predicting customer churn for a bank using machine learning. The goal is to identify customers who are likely to leave the bank, enabling proactive measures to retain them. The combination of several supervised model developed achieves an accuracy of 82% on the test data, with a precision of 0.73 and recall of 0.77. This metric suggest that the model is effective at identifying customers at risk of churning.

**Tools and Technologies**: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost.

## 1. Introduction

Customer churn is a critical metric for any business, particularly in the banking sector, where customer acquisition costs are high. Retaining existing customers is therefore a top priority. By predicting which customers are likely to churn, banks can take targeted actions to improve customer satisfaction and reduce churn rates.

## 2. Objective

The objective of this project is to build a machine learning model that accurately predicts customer churn. The predictions will help the bank to focus its retention efforts on high-risk customers, thereby reducing overall churn rates and increasing customer lifetime value.

## 3. Data Understanding

### 3.1 Data Sources

The data used in this project is a synthetic dataset provided by the bank, containing information about 10,000 customers. The dataset includes various features such as:

- Customer ID
- Surname
- Credit Score
- Geography (Country)
- Gender
- Age
- Tenure (Years with the bank)
- Balance
- Number of Products
- Has Credit Card
- IsActiveMember (whether the customer is active or not)
- Estimated Salary
- Exited (whether the customer has churned)

Data Source: [dataset-link](https://www.kaggle.com/datasets/barelydedicated/bank-customer-churn-modeling)

### 3.2 Data Preprocessing

The dataset required some preprocessing steps:
- **Handling Missing Values**: There were no missing values in the dataset.
- **Encoding Categorical Variables**: The categorical variables such as Gender and Geography were encoded using One-Hot Encoding.
- **Feature Scaling**: Features such as Balance and Estimated Salary were scaled using StandardScaler to ensure that they have the same scale.
- **Splitting the Data**: The data was split into a training set (60%), validation set (20%) and a test set (20%).

## 4. Exploratory Data Analysis (EDA)

### 4.1 Univariate Analysis

- **Age**: The age distribution shows that most customers are between 35 and 45 years old.
- **Balance**: Around 35% of customers have a balance close to zero.
- **Churn**: Approximately 20% of the customers in the dataset have churned.

### 4.2 Bivariate Analysis

- **Gender vs Churn**: Female customers are more likely to churn compared to male customers.
- **Geography vs Churn**: Customers from certain countries have a higher churn rate.
- **Tenure vs Churn**: Customers with shorter tenures are more likely to churn.
- **Number of products Vs Churn**: Customers with single product are more likely to churn.
- **Is Active Member vs Churn**: Inactive members seems to have more churn rate.


### 4.3 Correlation Analysis

Correlation analysis using chi-squared test was performed to understand the relationships between features. It was found that the correlation between most features was low, suggesting that multicollinearity is not a concern. Geography, Gender, Number of products and Activity of member shows correlation with Churn.

Data-exploration notebook can be found here: [data-exploration.ipynb](https://github.com/madhuri-15/Predicting-Churn-for-Bank-Customers/blob/main/Notebook/data-exploration.ipynb)

## 5. Feature Engineering
Based on exploratory data analysis, new features added based on various variables that includes age, balance, number of products, credit score which significantly improves the model performance.

Feature engingeering notebook can be found here: [feature-engineering.ipynb](https://github.com/madhuri-15/Predicting-Churn-for-Bank-Customers/blob/main/Notebook/feature-engineering.ipynb)

## 6. Model Development

### 6.1 Model Selection
The given dataset has imbalance class distribution as the number of customers churn were significantly lower than the customers who stayed. To handle imbalance class distribution, class weights in classifier were adjust accordingly.

Several machine learning models were considered for this task, including:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting Machine (GBM)
- Extreme Boosting (xgboost)

![classifier-performance](https://github.com/madhuri-15/Predicting-Churn-for-Bank-Customers/blob/main/Images/clf_performance.png)



### 6.2 Model Training

Each model was trained on the training dataset. Hyperparameter tuning was performed using GridSearchCV to find the best parameters for each model. The best model were selected based on cross-validation score on validation dataset. 
The models were evaluated based on roc-auc score, recall, and F1 score. 

Recall selected as the primary model evaluation metrics as we don't wanna missout any false negative(actual positive). Hence, precision-recall score is observed at various threshold, the final threshold is selected based on f1-score as it balances precision and recall.

|Model|fit_time|train_roc_auc|train_precision|train_recall|train_f1score|valid_roc_auc|valid_precision|valid_recall|valid_f1score|
|-|-|-|-|-|-|-|-|-|-|
|Gradient Boosting|2.859380	|0.875079	|0.832965	|0.725176	|0.760090	|0.857815	|0.814231	|0.710583	|0.743493
|XGBoost|1.849223	|0.883799	|0.838257	|0.725101	|0.761152	|0.856057	|0.817368	|0.711211	|0.744691
|Random Forest|0.606731	|0.885676	|0.733346	|0.794234	|0.753178	|0.850002	|0.711496	|0.765610	|0.728944
|KNN|0.056283	|0.877314	|0.821182	|0.697091	|0.732529	|0.846718	|0.810581	|0.700155	|0.733489
|Logistic Regression|0.040348	|0.847491	|0.701168	|0.768310	|0.717837	|0.836169	|0.697227	|0.766720	|0.713078


The ensemble of different classifiers were selected based on fitting time and evaluation metric score and evaluated on validation dataset. The final ensemble model includes Logistic Regression, K-Nearest, Random Forest, and Extreme Boosting classifiers.

Model selection notebook can be found here: [model-selection.ipynb](https://github.com/madhuri-15/Predicting-Churn-for-Bank-Customers/blob/main/Notebook/model-selection.ipynb)

### 6.3 Model Evaluation

After training, the models were evaluated on the test dataset. The selected final model is ensemble of several supervised classifiers whcih includes XGBoost, Random Forest, Logistic Regression, and KNN from grid-search best results.

This model performed the best, with the following metrics:

- ROC-AUC score: 86%
- Recall: 77%
- F1 Score: 75%

![test confusion matrix](https://github.com/madhuri-15/Predicting-Churn-for-Bank-Customers/blob/main/Images/image.png)

Model evaluation notebook can be found here: [model-evaluation.ipynb](https://github.com/madhuri-15/Predicting-Churn-for-Bank-Customers/blob/main/Notebook/model-evaluation.ipynb)


## 7. Result and Impact

The final model achiveved a accuracy of 85%, with  recall of 0.77. This indicates that the model was quite effective at identifying customers who are likely to reduced churn by 8%.

