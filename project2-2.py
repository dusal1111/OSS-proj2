#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def sort_dataset(df):
    return df.sort_values(by='year')

def split_dataset(df):
    df['salary'] *= 0.001

    train_df = df.iloc[:1718]
    test_df = df.iloc[1718:]
    
    X_train = train_df.drop('salary', axis=1)
    Y_train = train_df['salary']
    
    X_test = test_df.drop('salary', axis=1)
    Y_test = test_df['salary']
    
    return X_train, X_test, Y_train, Y_test

def extract_numerical_cols(df):
    return df[['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']]

def train_predict_decision_tree(X_train, Y_train, X_test):
    model = DecisionTreeRegressor(random_state=3)
    model.fit(X_train, Y_train)
    return model.predict(X_test)

def train_predict_random_forest(X_train, Y_train, X_test):
    model = RandomForestRegressor(random_state=3)
    model.fit(X_train, Y_train)
    return model.predict(X_test)

def train_predict_svm(X_train, Y_train, X_test):
    model = make_pipeline(StandardScaler(), SVR())
    model.fit(X_train, Y_train)
    return model.predict(X_test)

def calculate_RMSE(labels, predictions):
    return mean_squared_error(labels, predictions, squared=False)

if __name__ == '__main__':
    data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
    
    sorted_df = sort_dataset(data_df)    
    X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
    
    X_train = extract_numerical_cols(X_train)
    X_test = extract_numerical_cols(X_test)

    dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
    rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
    svm_predictions = train_predict_svm(X_train, Y_train, X_test)
    
    print("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))
    print("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))
    print("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))


# In[ ]:




