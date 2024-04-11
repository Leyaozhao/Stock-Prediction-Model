import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Reshape, Add, Multiply
from tensorflow.keras.models import Model
# coding=utf-8
import numpy as np
my_seed =64#16
np.random.seed(my_seed)
import random
random.seed(my_seed)
tf.random.set_seed(my_seed)
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from pylab import mpl
import matplotlib.pyplot as plt
#mpl.rcParams['font.sans-serif'] = ['SimHei']
#plt.rcParams['axes.unicode_minus'] = False
from keras.layers import Dense,Input, Flatten,Reshape,add, Multiply
from tensorflow.keras import layers, Model
import tushare as ts
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
#labels and negative signs are displayed normally
#plt.rcParams['font.sans-serif'] = ['SimHei']
#plt.rcParams['axes.unicode_minus'] = False
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#Data collection Data collection and storage in local csv file 601020
#Get the trading date, opening price, closing price, lowest price, highest price, 
#trading volume, price changes, and increase and decrease data of the stock in the past 300 days
data=ts.get_k_data('600519',start = '2023-03-01',end='2023-9-29')
print(data)

#Build training data
def train_data(data):
    dataset_train= data
    x= dataset_train[['close','high', 'low']]
    y = dataset_train[['open']]
    print(y.shape)
    s = MinMaxScaler(feature_range = (0, 1))
    sc = MinMaxScaler(feature_range = (0, 1))
    x = s.fit_transform(x)
    y = sc.fit_transform(y)
    t=5
    X_train = []
    y_train = []
    for i in range(t, 161):
        X_train.append(x[i-t :i, :])
        y_train.append(y[i, :])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train  = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],3))

    train_size=int(len(X_train)*0.8)
    X_train,X_test=X_train[0:train_size],X_train[train_size:len(X_train)]
    y_train,y_test=y_train[0:train_size],y_train[train_size:len(y_train)]
    return X_train,y_train,X_test,y_test,sc,s

t=5
m=3
#build the model
def stock_model(X_train, y_train):
    a = Input((t,m))  #input layer 
    x=tf.keras.layers.LSTM(32,return_sequences=True)(a)
    #x=tf.keras.layers.LSTM(32,return_sequences=True)(x)
    x = Flatten()(x)
    x = Dense(16,activation='relu')(x)
    x = Dense(1, activation="linear")(x)
    model = Model(inputs=a, outputs=x)
    model.compile(loss='mse',metrics=['mae'],optimizer='adam')
    #train the model
    #model.fit(X_train, y_train, epochs =200, batch_size =26,shuffle=True)
    history=model.fit(X_train, y_train, epochs =500, batch_size =26,validation_split=0.2,shuffle=True)
    #Plot loss graph
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title('LSTM_loss', fontsize='12')
    plt.ylabel('loss', fontsize='10')
    plt.xlabel('epoch', fontsize='10')
    plt.legend()
    plt.show()
    return model
    return model
def main():
    X_train, y_train,X_test,real_stock_price,sc,s = train_data(data)
    print(X_train.shape)
    model = stock_model(X_train, y_train)
    #fit the model
    predicted_stock_price = model.predict(X_test)

    real_stock_price = sc.inverse_transform(real_stock_price)
    #print(real_stock_price)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    #print(predicted_stock_price)
    #Predictive evaluation indicators
    # calculate MSE
    mse=mean_squared_error(predicted_stock_price,real_stock_price)
    # calculate RMSE
    rmse = sqrt(mean_squared_error(predicted_stock_price,real_stock_price))
    #calculate MAE
    mae=mean_absolute_error(predicted_stock_price,real_stock_price)
    #calculate R square
    r_square=r2_score(predicted_stock_price,real_stock_price)
    print('MSE: %.6f' % mse)
    print('RMSE: %.6f' % rmse)
    print('MAE: %.6f' % mae)
    print('R_square: %.6f' % r_square)

    plt.figure(figsize=(8, 4))
    plt.plot(real_stock_price[:,0], color='#1F77B4', label = 'actual closing price')
    plt.plot(predicted_stock_price[:,0], color = 'red', label = 'predict closing price')
    #plt.xticks(pd.date_range('2023-9-1','2023-10-30',freq='2d'), rotation = 50, fontsize = 6)
    plt.xlabel('Time')
    #plt.xlabel('Date', fontsize = 8)
    plt.ylabel('Closing price', fontsize = 8)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()