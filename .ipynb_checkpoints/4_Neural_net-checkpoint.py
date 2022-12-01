### Import libraries ----
import pandas as pd
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.utils import  plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os

# Set working directory
os.chdir('/Users/hannahfrank/ml_project')

### Load data ----
df_ucdp = pd.read_csv('ucdp_month.csv', 
                      index_col=0,
                      parse_dates=True)

### Prepare data -----

# Use 10 months sequences
number_s=11  
   
# Normalization
scaler = MinMaxScaler(feature_range=(0,1))
df = pd.DataFrame(scaler.fit_transform(df_ucdp))

# Create time sequences
ts_seq=[]
for col in range(len(df.columns)):
    for i in range(number_s-1,len(df)):
        ts_seq.append(df.iloc[i-number_s+1:i+1,col])
        
# Create training, validation and test partitions with 80-20-20 split
ts_seq=np.array(ts_seq)
ts_seq_l= ts_seq.reshape(len(df.columns),len(df.index)-number_s+1,number_s)

ts_seq_learn=ts_seq_l[:,:int(0.6*len(df)),:]
ts_seq_learn=ts_seq_learn.reshape(ts_seq_learn.shape[0]*ts_seq_learn.shape[1],number_s)

ts_seq_val=ts_seq_l[:,int(0.6*len(df)):int(0.8*len(df)),:]
ts_seq_val=ts_seq_val.reshape(ts_seq_val.shape[0]*ts_seq_val.shape[1],number_s)

ts_seq_test=ts_seq_l[:,int(0.8*len(df)):,:]
ts_seq_test=ts_seq_test.reshape(ts_seq_test.shape[0]*ts_seq_test.shape[1],number_s)

# Obtain input and output training, validation and test partitions
train_x = ts_seq_learn[:,:-1]
train_y = ts_seq_learn[:,-1]
val_x = ts_seq_val[:,:-1]
val_y = ts_seq_val[:,-1]
test_x = ts_seq_test[:,:-1]
test_y = ts_seq_test[:,-1]

### Hyperparameter tuning using validation data -----
param=[]
min_mse=np.inf

# Loop through different values of activation function, batch_size and dropout rate
for activ in ['relu','sigmoid']:
    for b_si in [30,50,100,200]:
        for drop in [0.25,0.5]:
            
            # Specify model
            model = keras.Sequential()
            model.add(Dense(int(number_s/2),input_dim=number_s-1,activation=activ))
            model.add(Dropout(drop))
            model.add(Dense(int(number_s/4),activation=activ))
            model.add(Dropout(drop))
            model.add(Dense(1,activation=activ))
            model.compile(optimizer='adam', loss='mse')
            
            # fit model, make predictions and calculate mse, wmse
            cb = keras.callbacks.EarlyStopping(monitor='loss', 
                                               patience=10,
                                               restore_best_weights=True)
            training = model.fit(x=train_x, 
                                 y=train_y, 
                                 batch_size=b_si, 
                                 epochs=100, 
                                 shuffle=True,
                                 callbacks=[cb],
                                 validation_data=(val_x, val_y))
            pred = model.predict(val_x)
            param.append([activ,
                          b_si,
                          drop,
                          mean_squared_error(val_y, pred),
                          mean_squared_error(val_y, pred, sample_weight=val_y+1)])
            
            # Keep predictions for model with smallest mse
            if mean_squared_error(val_y,pred)<min_mse:
                pred=model.predict(test_x)
                pred_fin=pred
                
# mse
mse = mean_squared_error(test_y,pred_fin)
print(mse)
                
# wmse
wmse =  mean_squared_error(test_y,pred_fin,sample_weight=test_y+1)
print(wmse)

# Plot architecture 
plot_model(model, to_file='model_plot.png', show_shapes=True)

# Save table for hyperparameter tuning
param=pd.DataFrame(param)
param.columns=['Activation','Batch_size','Dropout','MSE','WMSE']
param.to_latex('tuning_nn.tex', index=False)

### Convert predictions back to original dataframe format ----
df_nn = pred_fin.reshape((len(df.iloc[0,:]),int(len(ts_seq_test[:,:-1])/len(df.iloc[0,:]))))
df_nn = df_nn.T
df_nn = pd.DataFrame(df_nn)
df_nn = pd.DataFrame(scaler.inverse_transform(df_nn))
df_nn.columns = df_ucdp.columns
df_nn.index=df_ucdp.index[-70:]

# Save
df_nn.to_csv('preds_nn.csv', index=False)




