from src.utils import load_data,train_test
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.utils import save_model, normalize_data

df_ = load_data()
df = normalize_data(df_)

model_base_path = './model/'

X_train, X_test, y_train, y_test = train_test(df,'Churn')
lr = LogisticRegression()
rfc = RandomForestClassifier()

def logistic_train(X_train, y_train):
    model = lr.fit(X_train, y_train)
    save_model(model, 'LogisticRegression', model_base_path)   
    return model

def random_forest_train(X_train, y_train):
    model = rfc.fit(X_train, y_train)
    save_model(model, 'RandomForestClassifier', model_base_path)
    return model

def xgb_train(X_train, y_train):
    xgb_params = {
        'objective': 'binary:logistic',
        'max_depth': 6,  
        'learning_rate': 0.03,  
        'eval_metric': 'auc', 
        'subsample': 0.8,  
        'colsample_bytree': 0.8, 
        'seed': 42
    }   
    
    dtrain = xgb.DMatrix(data = X_train, label = y_train)
    
    model = xgb.train(xgb_params, dtrain, num_boost_round = 200)
    
    save_model(model,'Xgboost',model_base_path)
    
    return model
    
    
    

