
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt



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

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(100.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class AIModel_TF(nn.Module):
    def __init__(self):
        super(AIModel_TF, self).__init__()
        enc_input_dim=6
        dec_input_dim=1
        d_model=512
        nhead=8
        # hidden_dim=512
        # pos_dim=8
        self.enc_value_embedding = nn.Linear(enc_input_dim,d_model)#TokenEmbedding(c_in=enc_input_dim, d_model=d_model)
        self.dec_value_embedding = nn.Linear(dec_input_dim,d_model)  #TokenEmbedding(c_in=dec_input_dim, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # self.conv_1 = nn.Conv1d(6, 32, 3, stride=1, padding=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True,dropout=0.1,dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # self.fc_0 = nn.Linear(1,32)   
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True,dropout=0.1,dim_feedforward=512)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        # self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)  
        
        # self.fc_1 = nn.Linear(6,256)    
        self.fc_out = nn.Linear(512,4)     

        # self.dropout = nn.Dropout(0.02)

    def forward(self, input_F,input_Q):
        #B,T,F
        # print(input_F.shape,input_Q.shape)
        enc_embedded = self.enc_value_embedding(input_F)
        encoded = enc_embedded + self.position_embedding(input_F)
        dec_embedded = self.dec_value_embedding(input_Q)
        decoded = dec_embedded + self.position_embedding(input_Q)
        # print(input_F)
        
        encoded=self.transformer_encoder(encoded)
        decoded=self.transformer_decoder(decoded,encoded)

        x = self.fc_out(decoded)
        # print(x.shape,x[:,-4:,0].shape)
        return x


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
    data_train=pd.read_csv("../QoEdataall.csv",index_col=None).values
    return data_train[:,1:]

def feature_gen(data,F_idx,Q_idx,time_len,pre_len):
    feature_data=np.zeros((len(data)-pre_len,(len(F_idx)+len(Q_idx))*time_len))
    F_idx.append(Q_idx[0])
    for time_i in range(time_len,len(data)-pre_len):
        feature_data[time_i,:len(F_idx)*time_len]=data[time_i-time_len+1:time_i+1,F_idx].reshape(-1)
        # feature_data[time_i,len(F_idx)*time_len:]=data[time_i-time_len:time_i,Q_idx].reshape(-1)
                
    qoe_data=np.zeros((len(data)-pre_len,len(Q_idx)*pre_len))
    for time_i in range(time_len,len(data)-pre_len):
        qoe_data[time_i,:]=data[time_i:time_i+pre_len,Q_idx[0]]
    return feature_data[time_len:-pre_len,:],qoe_data[time_len:-pre_len,:]

class MyDataset(Dataset):
    def __init__(self, data_mat, qoe_mat,time_len,fea_len,pre_len):
        print(data_mat.shape)
        self.pre_len=pre_len
        self.feature=data_mat[:,:].astype(np.float32).reshape((-1,time_len,fea_len+1))
        # self.his_qoe=data_mat[:,6*time_len:].astype(np.float32).reshape((-1,time_len))
        # self.feature_all=np.
        
        self.label = qoe_mat.astype(np.float32).reshape((-1,1,self.pre_len))
        self.time_len=time_len
        self.fea_len = fea_len 
        self.len=len(self.feature)
        # print(self.feature.shape,self.label.shape)

        
    def __len__(self):
        return self.len

    def __getitem__(self, idx): 
        
        input_F = self.feature[idx,:,:-1].copy()
        input_Q=np.zeros((self.time_len+self.pre_len-1,1))
        input_Q[:time_len-1,0] = self.feature[idx,:-1,-1]
        label = self.label[idx,:,:].copy() 

          
        return (input_F,input_Q, label)
    
class MyTestDataset(Dataset):
    def __init__(self, data_mat, qoe_mat,time_len,fea_len,pre_len):
        print(data_mat.shape)
        self.pre_len=pre_len
        self.feature=data_mat[:,:].astype(np.float32).reshape((-1,time_len,fea_len+1))
        # self.his_qoe=data_mat[:,6*time_len:].astype(np.float32).reshape((-1,time_len))
        # self.feature_all=np.
        
        self.label = qoe_mat.astype(np.float32).reshape((-1,1,pre_len))
        self.time_len=time_len
        self.fea_len=fea_len
        self.len=len(self.feature)
        # print(self.feature.shape,self.label.shape)

        
    def __len__(self):
        return self.len

    def __getitem__(self, idx): 
        
        input_F = self.feature[idx,:,:-1].copy()
        input_Q=np.zeros((self.time_len+self.pre_len-1,1))
        input_Q[:time_len-1,0] = self.feature[idx,:-1,-1]
        label = self.label[idx,:,:].copy()   
        return (input_F,input_Q, label)


DEVICE=torch.device("cpu")
if torch.cuda.is_available():
        DEVICE=torch.device("cuda:0")  

if __name__ == '__main__':
    setup_seed(2023)
    data_train=load_qoe_data()
    for i in range(data_train.shape[1]):
        data_train[:,i]=max_min(data_train[:,i])
    F_idx=[0,1,2,3,4,5] 
    Q_idx=[6]
    time_len=8
    pre_len=5
    data_train, qoe_train = feature_gen(data_train,F_idx,Q_idx,time_len,pre_len)   
    print(data_train.shape,qoe_train.shape)
    num_train = int(len(data_train)*0.7)
    num_test = int(len(data_train)*0.3)
    data_test=data_train[-num_test:,:].copy()  
    data_train=data_train[:num_train,:].copy()
    qoe_test=qoe_train[-num_test:,:].copy() 
    qoe_train=qoe_train[:num_train,:].copy()  
    print(data_train.shape,qoe_train.shape)
    print(data_train[0,:].reshape((time_len,7)))
    # qoe_test=qoe_train.copy()
    # data_test=data_train.copy()

    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001
    TOTAL_EPOCHS = 100
    split_ratio = 0.95
    change_learning_rate_epochs = 50

    train_dataset = MyDataset(data_train,qoe_train,time_len,6,pre_len)
    train_loader = DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)  # shuffle 标识要打乱顺序
    
    test_dataset = MyTestDataset(data_test,qoe_test,time_len,6,pre_len)
    test_loader = DataLoader(dataset=test_dataset,
                                               batch_size=1,
                                               shuffle=False)  # shuffle 标识要打乱顺序
  
    y_pred_DNN=np.zeros((len(data_test),pre_len))
    y_pred_DNN_train=np.zeros((len(data_train),pre_len))
    model_save = 'modelSubmit_TF.pth' 
    #新建模型   
    model = AIModel_TF()
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    test_avg_min = 10000
    for epoch in range(TOTAL_EPOCHS):
        model.train()       
        # Learning rate decay
        if (epoch + 1) % 5 == 0:
            optimizer.param_groups[0]['lr'] *= 0.9 
            print('lr:%.4e' % optimizer.param_groups[0]['lr'])              
        #模型训练
        loss_avg = 0
        train_data_num=0
        for i, (input_F,input_Q,QoE) in enumerate(train_loader):

            input_F = input_F.float().to(DEVICE)
            input_Q = input_Q.float().to(DEVICE)
            QoE   = QoE.float().to(DEVICE)
            
            # 清零
            optimizer.zero_grad()
            
            #这里只用CSI，可以加入其他参数，主要看想怎么进行特征工程
            
            output = model(input_F,input_Q)[:,-pre_len:,0]
            loss = nn.L1Loss(reduction='sum')(output,QoE[:,0,:])
            # if (i%10==0):
            #     plt.plot(QoE[0,0,:])
            #     plt.plot(output[0,:].cpu().detach().numpy())
            #     plt.show()  
            loss.backward()
            optimizer.step()
            # print(loss)
            # y_pred_DNN_train[train_data_num:train_data_num+len(output),:]=output.cpu().detach().numpy()
            # train_data_num=train_data_num+len(output)
            loss_avg += loss.item() 

        loss_avg /= len(data_train)*4
        
        import matplotlib.pyplot as plt
        # plt.plot(qoe_train[:,0])
        # plt.plot(y_pred_DNN_train[:,0])
        # plt.show()          
        #验证集进行算法验证
        # model.eval()
        test_avg = 0
        test_data_num=0
        input_Q_pre=torch.zeros((1,time_len+pre_len,1)).float().to(DEVICE)
        for i, (input_F,input_Q,QoE) in enumerate(test_loader):
            model.eval()
    
            input_F = input_F.float().to(DEVICE)
            input_Q = input_Q.float().to(DEVICE)
            QoE   = QoE.float().to(DEVICE)
            if (i==0):
                input_Q_pre=input_Q.clone()
            
            # 清零
            output = model(input_F,input_Q_pre)[:,-pre_len:,0]
            y_pred_DNN[test_data_num:test_data_num+len(output),:]=output.cpu().detach().numpy()
            test_data_num=test_data_num+len(output)
            input_Q_pre[0,0:time_len-2,:]=input_Q_pre[0,1:time_len-1,:].clone()
            input_Q_pre[0,time_len-2,0]=output[0,0]
            input_Q_pre[0,time_len-1,0]=0
            # print(input_Q_pre)
            
            loss = nn.L1Loss(reduction='sum')(output,QoE[:,0,:])
            test_avg += loss.item() 
             
        test_avg /= len(data_test)*5
        
        # test_avg /= len(test_loader)
        print('Epoch : %d/%d, Loss: %.4f, Test: %.4f, BestTest: %.4f' % (epoch + 1, TOTAL_EPOCHS, loss_avg,test_avg,test_avg_min))
        if test_avg < test_avg_min:
            y_pred_DNN_best=y_pred_DNN.copy()
            print('Model saved!')
            test_avg_min = test_avg
            torch.save(model.state_dict(), model_save) 
            
            
            mae = mean_absolute_error(qoe_test, y_pred_DNN_best)
            mse = mean_squared_error(qoe_test, y_pred_DNN_best)
            print("预测结果：", y_pred_DNN_best)
            print("MAE：", mae,mse)            
            # import matplotlib.pyplot as plt
            # plt.plot(qoe_test[:,0])
            # plt.plot(y_pred_DNN[:,0])
            # plt.plot(y_pred_DNN[:,2])
            # plt.plot(y_pred_DNN[:,3])
            # plt.show()   