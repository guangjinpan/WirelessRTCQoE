
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error

def max_min(data):
    mindata=min(data)
    maxdata=max(data)
    if (mindata!=maxdata):
        data=(data - min(data)) / (max(data) - min(data))
    return data

def load_qoe_data():
    data_train=pd.read_csv("../data/QoEdata.csv",index_col=None).values
    return data_train[:,1:]

def feature_gen(data,F_idx,Q_idx,time_len,pre_len):
    feature_data=np.zeros((len(data)-pre_len,(len(F_idx)+len(Q_idx))*time_len))
    for time_i in range(time_len,len(data)-pre_len):
        feature_data[time_i,:len(F_idx)*time_len]=data[time_i-time_len:time_i,F_idx].reshape(-1)
        feature_data[time_i,len(F_idx)*time_len:]=data[time_i-time_len:time_i,Q_idx].reshape(-1)
                
    qoe_data=np.zeros((len(data)-pre_len,len(Q_idx)*pre_len))
    for time_i in range(time_len,len(data)-pre_len):
        qoe_data[time_i,:]=data[time_i:time_i+pre_len,Q_idx[0]]
    return feature_data[time_len:-pre_len,:],qoe_data[time_len:-pre_len,:]

if __name__ == '__main__':
    data_train=load_qoe_data()
    for i in range(data_train.shape[1]):
        data_train[:,i]=max_min(data_train[:,i])
    print(data_train.shape)
    F_idx=[0,1,2,3,4,5] 
    Q_idx=[6]
    time_len=8
    pre_len=4
    data_train, qoe_train = feature_gen(data_train,F_idx,Q_idx,time_len,pre_len)   
    print(data_train.shape,qoe_train.shape)
    num_train = int(len(data_train)*0.5)
    num_test = int(len(data_train)*0.4)
    data_test=data_train[-num_test:,:].copy()  
    data_train=data_train[:num_train,:].copy()
    qoe_test=qoe_train[-num_test:,:].copy() 
    qoe_train=qoe_train[:num_train,:].copy()  
    
    data_train=data_train[:,:time_len*6]
    data_test=data_test[:,:time_len*6]
    # data_train=np.hstack((data_train,qoe_train[:,1:]))
    # data_test=np.hstack((data_test,qoe_test[:,1:]))

    clf = RandomForestRegressor()

    # 拟合（训练）模型
    clf.fit(data_train, qoe_train)

    # 预测测试集
    y_pred_RandomForest = clf.predict(data_test)
    # y_pred_RandomForest[:]=0.515
    # 计算MAE
    mae = mean_absolute_error(qoe_test, y_pred_RandomForest)
    mse = mean_squared_error(qoe_test, y_pred_RandomForest)

    print("预测结果：", y_pred_RandomForest)
    print("MAE：", mae,mse)
    import matplotlib.pyplot as plt
    plt.plot(qoe_test[:,0])
    plt.plot(y_pred_RandomForest[:,0])
    plt.show()
    mae = mean_absolute_error(qoe_test[1:,:], qoe_test[:-1,:])
    mse = mean_squared_error(qoe_test[1:,:], qoe_test[:-1,:])
    print("MAE：", mae,mse)