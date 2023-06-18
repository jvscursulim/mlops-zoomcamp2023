#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import sys
import pandas as pd


# In[ ]:


year = int(sys.argv[1])
month = int(sys.argv[2])


# In[2]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[3]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[4]:


df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')


# In[5]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)
print(f"Duration mean predictions: {y_pred.mean()}")


# In[12]:


df["ride_id"] = f"{year:04d}/{month:02d}" + df.index.astype("str")


# In[14]:


df["predictions"] = y_pred.copy()


# In[15]:


df_result = df[["ride_id", "predictions"]].copy()


# In[16]:


output_file = f"predictions_yellow_{year:04d}_{month:02d}.parquet"


# In[17]:


df_result.to_parquet(output_file, engine="pyarrow", compression=None, index=False)

