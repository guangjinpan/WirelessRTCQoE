
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import random
import matplotlib.pyplot as plt


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

class AIModel_LSTM(nn.Module):
    def __init__(self):
        super(AIModel_LSTM, self).__init__()
        self.conv_1 = nn.Conv1d(1, 32, 3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.lstm = nn.LSTM(6, 64, 2, batch_first=True)  #(seq_length,batch_size,input_size)  
              
        self.fc_out = nn.Linear(64,1)     

        self.dropout = nn.Dropout(0.2)

    def forward(self, F):
        #B,T,F
        # print(F.shape)
        # F1 = F.reshape((len(F),-1,1,6))
        # x = self.conv_1(F1)
        # print(x.shape)
        # x = self.relu(x)
        # x = x.permute(0,2,1)
        # print(CSI_amp_ant_A.shape,CSI_amp_ant_B.shape)
        # F = F.permute(1,0,2)
        x, (hn, cn) = self.lstm(F)
        # x = x.reshape((-1,64*5))
        # x = torch.cat((x,F[:,30:]),dim=1)
        x = self.relu(x)
        x = self.fc_out(x)
        return x.reshape((-1))


class AIModel_DNN(nn.Module):
    def __init__(self):
        super(AIModel_DNN, self).__init__()
        
        self.relu   = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        
        self.fc_1 = nn.Linear(34,128)       
        self.fc_2 = nn.Linear(128,512)     
        self.fc_3 = nn.Linear(512,128)       
        self.fc_4 = nn.Linear(128,1)  
        self.dropout = nn.Dropout(0.2)

    def forward(self, F):
        #B,T,F
        # print(CSI_amp_ant_A.shape,CSI_amp_ant_B.shape)
        # F = F.permute(1,0,2)
        x = self.fc_1(F)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.relu(x)
        x = self.fc_3(x)
        x = self.relu(x)
        x = self.fc_4(x)
        return x




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


def feature_gen_lstm(data,F_idx,Q_idx,time_len):
    feature_data=np.zeros((len(data),len(F_idx)+len(Q_idx)))
    for time_i in range(len(data)):
        for F_i in range(len(F_idx)):
            feature_data[time_i,F_i]=data[time_i,int(F_idx[F_i])]
        for Q_i in range(len(Q_idx)):
            feature_data[time_i,Q_i+len(F_idx)]=data[time_i,int(Q_idx[Q_i])]
    return feature_data

class MyDataset(Dataset):
    def __init__(self, data_mat):
        print(data_mat.shape)

        
        
        self.feature=data_mat[:,:6].astype(np.float32)
        
        self.label = data_mat[:,6].astype(np.float32)
        self.len=1

        
    def __len__(self):
        return self.len

    def __getitem__(self, idx): 
        
        F = self.feature
        label = self.label     
        return (F, label)
    
class MyTestDataset(Dataset):
    def __init__(self, data_mat):
        self.feature=data_mat[:,:6].astype(np.float32)
        self.label = data_mat[:,6].astype(np.float32)
        self.len=1

        
    def __len__(self):
        return self.len

    def __getitem__(self, idx): 
        F = self.feature
        label = self.label     
        return (F, label)


DEVICE=torch.device("cpu")
if torch.cuda.is_available():
        DEVICE=torch.device("cuda:0")  

if __name__ == '__main__':
    setup_seed(2023)
    data_train,data_test=load_qoe_data()
    for i in range(data_train.shape[1]):
        data_train[:,i]=max_min(data_train[:,i])
        data_test[:,i]=max_min(data_test[:,i])
        
    F_idx=[2,4,6,7,8,9]
    Q_idx=[11]
    time_len=5
    data_train = feature_gen_lstm(data_train,F_idx,Q_idx,time_len)
    data_test = feature_gen_lstm(data_test,F_idx,Q_idx,time_len)
    print(data_train.shape)
    # data_train=np.hstack((data_train,qoe_train[:,1:]))
    # data_test=np.hstack((data_test,qoe_test[:,1:]))


    BATCH_SIZE = 1
    LEARNING_RATE = 0.002
    TOTAL_EPOCHS = 1000
    split_ratio = 0.95
    change_learning_rate_epochs = 50

    train_dataset = MyDataset(data_train)
    train_loader = DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)  # shuffle 标识要打乱顺序
    
    test_dataset = MyTestDataset(data_test)
    test_loader = DataLoader(dataset=test_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=False)  # shuffle 标识要打乱顺序
  
    y_pred_DNN=np.zeros((len(data_test),1))
    model_save = 'modelSubmit_LSTM.pth' 
    #新建模型   
    model = AIModel_LSTM()
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    test_avg_min = 10000
    for epoch in range(TOTAL_EPOCHS):
        model.train()       
        # Learning rate decay
        if (epoch + 1) % 200 == 0:
            optimizer.param_groups[0]['lr'] /= 2 
            print('lr:%.4e' % optimizer.param_groups[0]['lr'])              
        #模型训练
        loss_avg = 0
        for i, (Feature,QoE) in enumerate(train_loader):

            Feature = Feature.float().to(DEVICE)
            QoE   = QoE.float().to(DEVICE)
            
            # 清零
            optimizer.zero_grad()
            
            #这里只用CSI，可以加入其他参数，主要看想怎么进行特征工程
            
            output = model(Feature)
        
            loss = nn.L1Loss(reduction='mean')(output,QoE)
            loss.backward()
            optimizer.step()
            
            loss_avg += loss.item() 

        loss_avg /= len(train_loader)
        
        #验证集进行算法验证
        model.eval()
        test_avg = 0
        test_data_num=0
        for i, (Feature,QoE) in enumerate(test_loader):
    
            Feature = Feature.float().to(DEVICE)
            QoE   = QoE.float().to(DEVICE)
            
            # 清零
            optimizer.zero_grad()
            output = model(Feature)
            
            y_pred_DNN[test_data_num:test_data_num+len(output),0]=output.cpu().detach().numpy()
            test_data_num=test_data_num+len(output)
            
            loss = nn.L1Loss(reduction='sum')(output,QoE)
            test_avg += loss.item() 
            

        test_avg /= len(y_pred_DNN)
        
        # test_avg /= len(test_loader)
        print('Epoch : %d/%d, Loss: %.4f, Test: %.4f, BestTest: %.4f' % (epoch + 1, TOTAL_EPOCHS, loss_avg,test_avg,test_avg_min))
        if test_avg < test_avg_min:
            y_pred_DNN_best=y_pred_DNN.copy()
            print('Model saved!')
            test_avg_min = test_avg
            torch.save(model.state_dict(), model_save) 
            
            
    mae = mean_absolute_error(qoe_test[:,0], y_pred_DNN_best)
    print("预测结果：", y_pred_DNN_best)
    print("MAE：", mae)            
    import matplotlib.pyplot as plt
    plt.plot(qoe_test[:,0])
    plt.plot(y_pred_DNN_best[:,0])
    plt.show()   