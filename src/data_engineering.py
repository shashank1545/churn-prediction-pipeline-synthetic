import pandas as pd

df = pd.read_excel('./data/raw/customer_churn_large_dataset.xlsx')

def features(df):
    # Removing columns Name and CustomerID as they are unique for everyone
    df.drop(columns=['Name','CustomerID'],axis=1,inplace=True,errors='ignore')

    # Encoding Gender and Location
    df['Gender'] = df['Gender'].map({'Female':0,'Male':1})
    df['Location'] = df['Location'].map({'Chicago':0,'Houston':1,'Los Angeles':2,'Miami':3,'New York':4})

    # Avg monthly data usage = Total / Months
    df['Avg_Usage_GB']  = df['Total_Usage_GB'] / df['Subscription_Length_Months']

    # Cost per gb = Monthly bill * moths / total gb
    df['Cost_Per_GB'] = (df['Subscription_Length_Months'] * df['Monthly_Bill']) / df['Total_Usage_GB']

    return df

df_m = features(df)

# Saving as parquet for faster read time
df_m.to_parquet('./data/processed/customer_churn_large_dataset.parquet')

