from src.train import logistic_train, random_forest_train, xgb_train
from src.utils import load_data, train_test, calculate_roc, normalize_data, preprocess
import os
import pickle
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np


df = normalize_data(load_data())

X_train, X_test, y_train, y_test = train_test(df,'Churn')

def model_predict(name:str, X_train, X_test, y_train, y_test) :
    if (name == "LogisticRegression"):
        if os.path.exists(f'./model/{name}_model.pkl') :
            with open (f'./model/{name}_model.pkl','rb') as f:
                model = pickle.load(f)
                calculate_roc(model,  X_train, X_test, y_train, y_test)
        else:
            model = logistic_train(X_train, y_train)
            calculate_roc(model,  X_train, X_test, y_train, y_test)
    elif (name == "RandomForestClassifier"):
        if os.path.exists(f'./model/{name}_model.pkl') :
            with open (f'./model/{name}_model.pkl','rb') as f:
                model = pickle.load(f)
                calculate_roc(model,  X_train, X_test, y_train, y_test)
        else:
            model = random_forest_train(X_train, y_train)
            calculate_roc(model,  X_train, X_test, y_train, y_test)
    elif (name == "Xgboost"):
        if os.path.exists(f'./model/{name}_model.pkl') :
            with open (f'./model/{name}_model.pkl','rb') as f:
                model = pickle.load(f)
                dtrain = xgb.DMatrix(X_train)
                dtest = xgb.DMatrix(X_test)
                y_pred_train = model.predict(dtrain)
                y_pred_test = model.predict(dtest)
                
                train_auc = roc_auc_score(y_train, y_pred_train)
                test_auc = roc_auc_score(y_test, y_pred_test)
                print(f"Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f} for Xgboost()")
                
        else:
            model = xgb_train(X_train, y_train)
            dtrain = xgb.DMatrix(data = X_train, label = y_train)
            dtest = xgb.DMatrix(data = X_test, label = y_test)
            y_pred_train = model.predict(dtrain)
            y_pred_test = model.predict(dtest)
            train_auc = roc_auc_score(y_train, y_pred_train)
            test_auc = roc_auc_score(y_test, y_pred_test)
            print(f"Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f} for Xgboost()")
            
    else:
        print("Model is not defined for this dataset.")

def predict(x_test:pd.DataFrame,name='Xgboost'):
    X_test = preprocess(x_test,test=True)
    if (name == "LogisticRegression"):
        with open(f'./model/{name}_model.pkl','rb') as f:
            model  = pickle.load(f)
        y_test_pred = model.predict(X_test)
    elif (name == "RandomForestClassifier"):
        with open(f'./model/{name}_model.pkl','rb') as f:
            model  = pickle.load(f)
        y_test_pred = model.predict(X_test)
    elif (name == 'Xgboost'):
        with open(f'./model/{name}_model.pkl','rb') as f:
            model  = pickle.load(f)
        dtest = xgb.DMatrix(X_test)
        y_test_pred = model.predict(dtest)
    else :
        y_test_pred = "Could not find model implemented for dataset."
        
    return y_test_pred
