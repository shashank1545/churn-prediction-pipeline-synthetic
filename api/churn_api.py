# app/churn_api.py

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, validator
import pandas as pd
from src.inference import predict

app = FastAPI(title="Customer Churn Prediction API")

# Define request schema
class CustomerData(BaseModel):
    Age: float = Field(..., gt=15, lt=75, description="Age of the customer")
    Gender: str = Field(..., pattern="^(Male|Female)$", description="Gender of the customer")
    Location: str = Field(..., pattern="^(Chicago|Houston|Los Angeles|Miami|New york)$", description="Location of the customer")
    Subscription_Length_Months: float = Field(...,gt=1 ,lt=24, description="Length of subscription in months")
    Monthly_Bill: float = Field(..., gt=30,lt=100, description="Monthly bill amount")
    Total_Usage_GB: float = Field(..., gt=50,lt=500, description="Total usage in GB")
    Model: str = Field(..., pattern="^(LogisticRegression|RandomForestClassifier|Xgboost)$", description="Model for prediction")

    class Config:
        schema_extra = {
            "example": {
                "Age": 30.0,
                "Gender": "Male",
                "Location": "Chicago",
                "Subscription_Length_Months": 12.0,
                "Monthly_Bill": 50.0,
                "Total_Usage_GB": 100.0,
                "Model":"LogisticRegression"
            }
        }
    
@app.post("/predict")
async def predict_churn(req: Request):
    data = await req.json()
    
    df = pd.DataFrame(data,index=[0])
    
    # Get the model name and drop it from dataframe
    model = df['Model']
    model_name = model.values[0]
    df_copy = df.drop(columns='Model', axis=1, errors='ignore')
    
    # Calculate probaility
    probability = predict(df_copy,model_name)
       
    return {"Churn prediction": probability.item()}
