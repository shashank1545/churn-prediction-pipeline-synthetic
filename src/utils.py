import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold
import xgboost as xgb
import pickle
from src.data_engineering import features


def load_data(path='./data/processed/customer_churn_large_dataset.parquet') -> pd.DataFrame :
    return pd.read_parquet(path)

def normalize_data(df:pd.DataFrame,test=False) :
    scaler = MinMaxScaler()
    df_train = load_data()
    df_train_fit = scaler.fit(df_train)
    if not test:
        np_scaled = df_train_fit.transform(df_train)
        df_scaled = pd.DataFrame(np_scaled,columns = df_train.columns)
    else:
        df_test = df_train.drop(columns=['Churn'],axis=1)
        df_test_fit = scaler.fit(df_test)
        np_scaled = df_test_fit.transform(df)
        df_scaled = pd.DataFrame(np_scaled,columns = df_test.columns)
        
    return df_scaled

def evaluate_model(model,x_test,y_test) :
    preds = model.predict(x_test)
    auc = roc_auc_score(y_test, preds)
    print(f"Test AUC: {auc:.4f}")
    
def train_test(df:pd.DataFrame,col:str) :
    y = df[col]
    X = df.drop(columns=[col])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train,X_test,y_train,y_test

# def k_fold_training(model, X_train, y_train) :
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     train_auc_score = []
#     val_auc_score = []
    
#     for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
#         X_train_fold = X_train.iloc[train_idx]
#         y_train_fold = y_train.iloc[train_idx]
#         X_val_fold = X_train.iloc[val_idx]
#         y_val_fold = y_train.iloc[val_idx]
        
#         dtrain_fold = xgb.DMatrix(X_train_fold, label=y_train_fold)
#         dval_fold = xgb.DMatrix(X_val_fold, label=y_val_fold)
        
#         model = xgb_train(dtrain_fold, dval_fold)
        
#         y_pred_train = model.predict(dtrain_fold)
#         y_pred_val = model.predict(dval_fold)

#         train_auc = roc_auc_score(y_train_fold, y_pred_train)
#         val_auc = roc_auc_score(y_val_fold, y_pred_val)

#         train_auc_score.append(train_auc)
#         val_auc_score.append(val_auc)

#         print(f"Fold {fold+1}, Train AUC: {train_auc:.4f}, Validation AUC: {val_auc:.4f}")
            
def save_model(model,name:str,path) : 
    with open(f'{path}/{name}_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
def calculate_roc(model, X_train, X_test, y_train, y_test):
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_auc = roc_auc_score(y_train, y_pred_train)
    test_auc = roc_auc_score(y_test, y_pred_test)
    print(f"Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f} for {model}")
    
def preprocess(df:pd.DataFrame,test=True):
    if not test:
        print("Train already processed!!")
        return False
    df_mod = features(df)
    df_mod_norm = normalize_data(df_mod,test)
    return df_mod_norm