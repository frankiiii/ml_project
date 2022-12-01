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
ts_seq_train = ts_seq_learn[:,:int(tr_test_split*len(ts_seq_learn)),:]
ts_seq_val = ts_seq_learn[:,int(tr_test_split*len(ts_seq_learn)):,:]
ts_seq_learn=ts_seq_learn.reshape(ts_seq_learn.shape[0]*ts_seq_learn.shape[1],number_s)
ts_seq_train=ts_seq_train.reshape(ts_seq_train.shape[0]*ts_seq_train.shape[1],number_s)
ts_seq_val=ts_seq_val.reshape(ts_seq_val.shape[0]*ts_seq_val.shape[1],number_s)
ts_seq_test=ts_seq_l[:,int(tr_test_split*len(df)):,:]
ts_seq_test=ts_seq_test.reshape(ts_seq_test.shape[0]*ts_seq_test.shape[1],number_s)
# Creation of input/output for each set
train_x = ts_seq_learn[:,:-1]
train_y = ts_seq_learn[:,-1]
val_x = ts_seq_val[:,:-1]
val_y = ts_seq_val[:,-1]
test_x = ts_seq_test[:,:-1]
test_y = ts_seq_test[:,-1] 

param=[]
min_mse=np.inf
for activ in ['relu','sigmoid']:
    for b_si in [30,50,100,200]:
        for drop in [0.25,0.5]:
            #creation of the model
            model = keras.Sequential()
            model.add(Dense(int(number_s/2),input_dim=number_s-1,activation=activ))
            model.add(Dropout(drop))
            model.add(Dense(int(number_s/4),activation=activ))
            model.add(Dropout(drop))
            model.add(Dense(1,activation=activ))
            model.compile(optimizer='adam', loss='mse')
            
            # Learning/test of the model
            cb = keras.callbacks.EarlyStopping(monitor='loss', patience=5,restore_best_weights=True)
            training = model.fit(x=train_x, y=train_y, batch_size=b_si, epochs=50, shuffle=True,callbacks=[cb],validation_data=(val_x,val_y))
            pred = model.predict(test_x)
            param.append([activ,b_si,drop,mean_squared_error(test_y,pred),mean_squared_error(test_y,pred,sample_weight=test_y+1)])
            if mean_squared_error(test_y,pred)<min_mse:
                pred_fin=pred
                min_mse = mean_squared_error(test_y,pred)
                weighted_mse =  mean_squared_error(test_y,pred,sample_weight=test_y+1)

df_nn = pred_fin.reshape((len(df.iloc[0,:]),int(len(ts_seq_test[:,:-1])/len(df.iloc[0,:]))))
df_nn = df_nn.T
df_nn = pd.DataFrame(df_nn)
df_nn = pd.DataFrame(scaler.inverse_transform(df_nn))
df_nn.columns = df_tot.columns
df_nn.index=df_tot.index[-70:]
df_nn.to_csv('preds_nn.csv', index=False)



