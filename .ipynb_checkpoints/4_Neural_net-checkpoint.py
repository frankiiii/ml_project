# -*- coding: utf-8 -*-
"""
Created on Monday Nov 28 14:07:24 2022
@author: Thomas
"""
import pandas as pd
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

df_tot= pd.read_csv('ucdp_month.csv',index_col=0,parse_dates=True)
number_s=11          # 10 months sequences
tr_test_split=0.8    # 80% train - 20% test

# Normalization
scaler = MinMaxScaler(feature_range=(0,1))
df= pd.DataFrame(scaler.fit_transform(df_tot))

# Creation of the sequences
ts_seq=[]
for col in range(len(df.columns)):
    for i in range(number_s-1,len(df)):
        ts_seq.append(df.iloc[i-number_s+1:i+1,col])
# Creation of trainset and testet
ts_seq=np.array(ts_seq)
ts_seq_l= ts_seq.reshape(len(df.columns),len(df.index)-number_s+1,number_s)
ts_seq_learn=ts_seq_l[:,:int(tr_test_split*len(df)),:]
ts_seq_learn=ts_seq_learn.reshape(ts_seq_learn.shape[0]*ts_seq_learn.shape[1],number_s)
ts_seq_test=ts_seq_l[:,int(tr_test_split*len(df)):,:]
ts_seq_test=ts_seq_test.reshape(ts_seq_test.shape[0]*ts_seq_test.shape[1],number_s)
# Creation of input/output for each set
train_x = ts_seq_learn[:,:-1]
train_y = ts_seq_learn[:,-1]
test_x = ts_seq_test[:,:-1]
test_y = ts_seq_test[:,-1] 
#creation of the model
model = keras.Sequential()
model.add(Dense(int(number_s/2),input_dim=number_s-1,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(int(number_s/4),activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam', loss='mse')
# Learning/test of the model
training = model.fit(x=train_x, y=train_y, batch_size=30, epochs=50, shuffle=True)
pred = model.predict(test_x)
mse = mean_squared_error(test_y,pred)
weighted_mse =  mean_squared_error(test_y,pred,sample_weight=test_y+1)
