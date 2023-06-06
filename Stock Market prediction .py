#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas_datareader as pdr 
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 


# In[ ]:


df=pdr.get_data_tiingo('AAPL',api_key='9b2813b091e43090b9084a9fe060b5b2ffb54ba1')


# In[ ]:


df.to_csv('Apple.csv')


# In[ ]:


df=pd.read_csv('Apple.csv')


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df1=df.reset_index()['open']


# In[ ]:


df1


# In[ ]:


plt.plot(df1)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler 
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[ ]:


df1


# In[ ]:


training_size=int(len(df1)*0.80)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size:],df1[training_size:len(df1),:1]


# In[ ]:


print(train_data)


# In[ ]:


import numpy 
def create_dataset(dataset,time_step=1):
    dataX, dataY=[], [] 
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return numpy.array(dataX),numpy.array(dataY)


# In[ ]:


time_step=100
X_train,Y_train=create_dataset(train_data,time_step)
X_test,y_test=create_dataset(test_data,time_step)


# In[ ]:


print(X_train.shape),print(Y_train.shape)


# In[ ]:


print(X_test.shape),print(y_test.shape)


# In[ ]:


X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)


# In[ ]:


from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import LSTM 


# 

# 

# In[ ]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[ ]:


model.summary()


# In[ ]:


model.fit(X_train,Y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64,verbose=1)


# In[ ]:


train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[ ]:


train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[ ]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(Y_train,train_predict))


# In[ ]:


math.sqrt(mean_squared_error(y_test,test_predict))


# In[ ]:


look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[ ]:


len(test_data)


# In[ ]:


x_input=test_data[152:].reshape(1,-1)
x_input.shape


# In[ ]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[ ]:


temp_input


# In[ ]:


from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<100):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[ ]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


len(df1)


# In[ ]:


plt.plot(day_new,scaler.inverse_transform(df1[1157:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[ ]:


df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])


# In[ ]:




