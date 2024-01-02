
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def max_min(data):
    mindata=min(data)
    maxdata=max(data)
    if (mindata!=maxdata):
        data=(data - min(data)) / (max(data) - min(data))
    return data

def load_qoe_data():
    data_train=np.zeros((1,20))
    data_test=np.zeros((1,20))
    for i in range(1,8):
        j=1
        file_name1 =f'../data/UE2_{i}_{j}_train.csv'
        datax=pd.read_csv(file_name1,index_col=0).values  
        data_train=np.vstack((data_train,datax))
        file_name1 =f'../data/UE2_{i}_{2}_test.csv'
        datax=pd.read_csv(file_name1,index_col=0).values  
        data_test=np.vstack((data_test,datax))    
    data_train=data_train[1:,:]
    data_test=data_test[1:,:]

    return data_test,data_train

def feature_gen(data,F_idx,Q_idx,time_len):
    feature_data=np.zeros((len(data),len(F_idx),time_len))
    for time_i in range(time_len,len(data)):
        for F_i in range(len(F_idx)):
            for time_j in range(time_len):
                feature_data[time_i,F_i,time_j]=data[time_i-time_j,int(F_idx[F_i])]
                
    qoe_data=np.zeros((len(data),len(Q_idx),time_len))
    for time_i in range(time_len,len(data)):
        for Q_i in range(len(Q_idx)):
            for time_j in range(time_len):
                qoe_data[time_i,Q_i,time_j]=data[time_i-time_j,int(Q_idx[Q_i])]
    return feature_data[time_len:,:,:].reshape((len(data)-time_len,-1)),qoe_data[time_len:,:,:].reshape((len(data)-time_len,-1))

if __name__ == '__main__':
    data_train,data_test=load_qoe_data()
    for i in range(data_train.shape[1]):
        data_train[:,i]=max_min(data_train[:,i])
        data_test[:,i]=max_min(data_test[:,i])
        
    F_idx=[2,4,6,7,8,9]
    Q_idx=[11]
    time_len=5
    data_train, qoe_train = feature_gen(data_train,F_idx,Q_idx,time_len)
    data_test, qoe_test = feature_gen(data_test,F_idx,Q_idx,time_len)
    # data_train=np.hstack((data_train,qoe_train[:,1:]))
    # data_test=np.hstack((data_test,qoe_test[:,1:]))

    clf = RandomForestRegressor()

    # 拟合（训练）模型
    clf.fit(data_train[:,:], qoe_train[:,0])

    # 预测测试集
    y_pred_RandomForest = clf.predict(data_test[:,:])

    # 计算MAE
    mae = mean_absolute_error(qoe_test[:,0], y_pred_RandomForest)

    print("预测结果：", y_pred_RandomForest)
    print("MAE：", mae)
    import matplotlib.pyplot as plt
    plt.plot(qoe_test[:,0])
    plt.plot(y_pred_RandomForest)
    plt.show()