#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 23:39:14 2020

@author: danijelmisulic
"""
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
import xgboost as xgb


def get_to_know_data(data):
    #Getting to know the data
    print ("Number of observations:", data.shape[0])
    print ("Number of columns:", data.shape[1])
    print ("Column names", data.columns)
    
    #Checking first five raws
    print(data.head())
    #Checking if there are null values, and dtype for every column
    print(data.info())
    #Additional null values check
    print (data.isnull().sum())
    
    #Looking into descriptive statistics
    print (data.describe())
    print(data.skew())
    
    #Checking if output classes are balanced
    class_dist = data.groupby('sleep_state').size()
    print (class_dist)
    
    
def data_visualizations(data):
    visualize_class_distribution(data)
    visualize_skewness(data)
    boxplot_visualizations(data)
    heatmap_visualizations(data)
    
    
def visualize_class_distribution(data):
    class_dist=data.groupby('sleep_state').size()
    class_label=pd.DataFrame(class_dist,columns=['Size'])
    plt.figure(figsize=(8,6))
    sns.barplot(x=class_label.index,y='Size',data=class_label)


def visualize_skewness(data):
    skew=data.skew()
    skew_df=pd.DataFrame(skew,index=None,columns=['Skewness'])
    plt.figure(figsize=(15,7))
    sns.barplot(x=skew_df.index,y='Skewness',data=skew_df)
    plt.xticks(rotation=90)
    plt.show()
    
    
def boxplot_visualizations(data):
    
    for i, col in enumerate(data.columns):
        plt.figure(i,figsize=(8,4))
        sns.boxplot(x=data['sleep_state'], y=col, data=data, palette="coolwarm")
    
 
def heatmap_visualizations(data):
    plt.figure(figsize=(15,8))
    sns.heatmap(data.corr(),cmap='magma',linecolor='white',linewidths=1,annot=True)
    

def prepare_data_for_modeling(data, X):
    Y = data['sleep_state']
    
    #split training and testing data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
    return x_train, x_test, y_train, y_test


def select_features(data):
    #Split X and Y variables
    Y = data['sleep_state']
    X = data.drop(["sleep_state"], axis=1, inplace=False)
    
    ETC = ExtraTreesClassifier()
    ETC = ETC.fit(X, Y)

    model = SelectFromModel(ETC, prefit=True)
    X = model.transform(X)
    return X


def fitting_model(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    model_pred = model.predict(x_test)
    model_accuracy = accuracy_score(model_pred , y_test)
    
    return model_accuracy


def create_decision_tree_model(x_train, x_test, y_train, y_test):    
    dectree = DecisionTreeClassifier()
    accuracy = fitting_model(dectree, x_train, x_test, y_train, y_test)
    print (accuracy)
    
    
def create_random_forest_model(x_train, x_test, y_train, y_test):    
    randfor = RandomForestClassifier()
    accuracy = fitting_model(randfor, x_train, x_test, y_train, y_test)
    print (accuracy)
    

def create_XGBoost_model(x_train, x_test, y_train, y_test):
    xg_reg = xgb.XGBClassifier(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
    accuracy = fitting_model(xg_reg, x_train, x_test, y_train, y_test)
    print (accuracy)
    
    
def create_logistic_regresion_model(x_train, x_test, y_train, y_test):    
    logreg = LogisticRegression(solver='liblinear', multi_class='ovr')
    accuracy = fitting_model(logreg, x_train, x_test, y_train, y_test)
    print (accuracy)


def create_KNN_model(x_train, x_test, y_train, y_test):
    #Setup arrays to store training and test accuracies
    neighbors = np.arange(1,7)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
    
    for i, k in enumerate(neighbors):
        #Setup a classifier with k neighbors
        knn = KNeighborsClassifier(n_neighbors=k)
        
        #Fit the model on training and testing data
        knn.fit(x_train, y_train)
        
        #Compute accuracy on the training set
        train_accuracy[i] = knn.score(x_train, y_train)
        
        #Compute accuracy on the test set
        test_accuracy[i] = knn.score(x_test, y_test) 
        
    print (train_accuracy)
    print (test_accuracy)

    
if __name__ == "__main__":        
    data = pd.read_csv('Task_DS_BEG_nightly_data (1).csv')
    data.drop(["Unnamed: 0", "app_frame_no", "time_stamp"], axis=1, inplace=True)
    #get_to_know_data(data)
    #data_visualizations(data)
    
    #all_features_X = data.drop(["sleep_state"], axis=1, inplace=False)
    #x_train, x_test, y_train, y_test = prepare_data_for_modeling(data, all_features_X)
    
    #returns activity_index, perfusion_index_green, perfusion_index_infrared, temp_amb, temp_skin
    preselected_features_X = select_features(data)
    x_train, x_test, y_train, y_test = prepare_data_for_modeling(data, preselected_features_X)
    
    create_decision_tree_model(x_train, x_test, y_train, y_test)
    create_XGBoost_model(x_train, x_test, y_train, y_test)
    create_random_forest_model(x_train, x_test, y_train, y_test)
    create_logistic_regresion_model(x_train, x_test, y_train, y_test)  
    create_KNN_model(x_train, x_test, y_train, y_test)
    

    
    
    
    
    
    
    
    
    