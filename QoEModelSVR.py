
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def smape(y_true,y_pred):
    n = len(y_true)
    np.seterr(divide='ignore', invalid='ignore')
    sum = np.abs((y_true - y_pred) / (y_true + y_pred))
    smape = np.mean(sum)*2
    return smape

def max_min(data):
    mindata=min(data)
    maxdata=max(data)
    if (mindata!=maxdata):
        data=(data - mindata) / (maxdata - mindata)
    return data,mindata,maxdata

def max_min_rec(data,mindata,maxdata):
    if (mindata!=maxdata):
        data = data * (maxdata - mindata) + mindata
    return data

def load_qoe_data():
    data_train=pd.read_csv("../QoEdataall.csv",index_col=None).values
    return data_train[:,1:]

def feature_gen(data,F_idx,Q_idx,time_len,pre_len):
    feature_data=np.zeros((len(data)-pre_len,(len(F_idx))*time_len))
    for time_i in range(time_len,len(data)-pre_len):
        feature_data[time_i,:len(F_idx)*time_len]=data[time_i-time_len+1:time_i+1,F_idx].reshape(-1)
        # feature_data[time_i,len(F_idx)*time_len:]=data[time_i-time_len:time_i,Q_idx].reshape(-1)
                
    qoe_data=np.zeros((len(data)-pre_len,len(Q_idx)*pre_len))
    for time_i in range(time_len,len(data)-pre_len):
        qoe_data[time_i,:]=data[time_i:time_i+pre_len,Q_idx[0]]
    return feature_data[time_len:-pre_len,:],qoe_data[time_len:-pre_len,:]

if __name__ == '__main__':
    data_all=load_qoe_data()
    data_scale=np.zeros((data_all.shape[1],2))
    for i in range(data_all.shape[1]):
        data_all[:,i],data_scale[i,0],data_scale[i,1]=max_min(data_all[:,i])
        
    print(data_scale)
    F_idx=[0,1,2,3,4,5] 
    Q_idx=[6]
    
    
    time_len=8
    for time_len in range(2,20,2):
        pre_len=5
        data_train, qoe_train = feature_gen(data_all,F_idx,Q_idx,time_len,pre_len)   
        num_train = int(len(data_train)*0.7)
        num_test = int(len(data_train)*0.3)
        data_test=data_train[-num_test:,:].copy()  
        data_train=data_train[:num_train,:].copy()
        qoe_test=qoe_train[-num_test:,:].copy() 
        qoe_train=qoe_train[:num_train,:].copy()  


        clf = SVR(kernel='linear')

        # 拟合（训练）模型
        data_train=data_train[:,:time_len*6]
        data_test=data_test[:,:time_len*6]    

        # 预测测试集
        y_pred_RandomForest = qoe_test.copy()
        for i in range(pre_len):
            clf.fit(data_train, qoe_train[:,i])
            y_pred_RandomForest[:,i] = clf.predict(data_test)

        # 计算MAE
        qoe_test=max_min_rec(qoe_test,data_scale[Q_idx[0],0],data_scale[Q_idx[0],1])
        y_pred_RandomForest=max_min_rec(y_pred_RandomForest,data_scale[Q_idx[0],0],data_scale[Q_idx[0],1])

        mae_res = mean_absolute_error(qoe_test, y_pred_RandomForest)
        mse_res = mean_squared_error(qoe_test, y_pred_RandomForest)
        smape_res= smape(qoe_test, y_pred_RandomForest)
        r2_res=r2_score(qoe_test, y_pred_RandomForest)
        # print("预测结果：", time_len,y_pred_RandomForest)
        print("MAE:", time_len,mae_res,mse_res,smape_res,r2_res)
    # import matplotlib.pyplot as plt
    # plt.plot(qoe_test)
    # plt.plot(y_pred_RandomForest)
    # plt.show()