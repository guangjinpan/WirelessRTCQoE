
import os
import re
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_json_files(directory,UE_i):
    data=np.zeros((100000,20))
    files = os.listdir(directory)  # 获取目录下的所有文件

    json_files = []
    files=sorted(files)
    for file_name in files:
#         print(file_name)
        # 使用正则表达式匹配文件名称
        if UE_i==1:
            pattern = r'0-\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}.\d{3}\.json'
        if UE_i==2:
            pattern = r'1-\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}.\d{3}\.json'
        if re.match(pattern, file_name):
            json_files.append(file_name)

    cnt=0
            
            
    for json_file in json_files:
        # 读取每个json文件的内容
        date_string = json_file[2:24]
        date_format = "%Y-%m-%d_%H_%M_%S.%f"
        timestamp = datetime.datetime.strptime(date_string, date_format).timestamp()                       
        data[cnt,0]=timestamp
        
        with open(os.path.join(directory, json_file), 'r') as file:
            json_data = json.load(file)
            # print(json_data)
            #0time/1rnti/2mcs/3sdubyte/4prb/5pdu/6harq/10UEidx/11bitrate/12rtt/13pkts/14payload/15loss
            data[cnt,10]=int(json_file[0])+10
            data[cnt,11]=json_data['delayTargetBitrate']
            data[cnt,12]=json_data['rtt']
            data[cnt,13]=json_data['sendpkts']
            data[cnt,14]=json_data['payload_total']
            data[cnt,15]=json_data['averageLoss']
        cnt=cnt+1

           
    return(data[:cnt,:])


def read_json_files_oai(directory,UE_i):
    data_oai=np.zeros((100000,20))
    files = os.listdir(directory)  # 获取目录下的所有文件

    json_files = []
    files=sorted(files)
    for file_name in files:
#         print(file_name)
        # 使用正则表达式匹配文件名称
        if UE_i==1:
            pattern = r'\d{4}-\d{2}-\d{2} \d{2}_\d{2}_\d{2}.\d{1}\.json'

        if re.match(pattern, file_name):
            json_files.append(file_name)

    cnt=0
            
#     print(json_files)

    for json_file in json_files:
        # 读取每个json文件的内容
#         date_string = json_file[0:21]
#         date_format = "%Y-%m-%dT%H:%M:%S.%f"
#         timestamp = datetime.datetime.strptime(date_string, date_format).timestamp()                       
#         data[cnt,0]=timestamp
        # print(json_files)
        with open(os.path.join(directory, json_file), 'r') as file:
            json_data = json.load(file)
            # print(json_file)
            data1=json_data["mac_stats"][0]
            for rnti_i in range(1):
                date_string = json_data["date_time"]
                date_format = "%Y-%m-%dT%H:%M:%S.%f"
                timestamp = datetime.datetime.strptime(date_string, date_format).timestamp()                
                
                data2=data1["ue_mac_stats"][rnti_i]
                data3=data2["mac_stats"]
                # print(data3["rrcMeasurements"])
                # print(data3["macStats"])
                 
                data4=data3["macStats"]
                data_oai[cnt,0]=timestamp
                data_oai[cnt,1]=data3["rnti"]
                
                data_oai[cnt,2]=data4["mcs1Dl"]
                data_oai[cnt,3]=data4["totalBytesSdusDl"]
                data_oai[cnt,4]=data4["totalPrbDl"]
                data_oai[cnt,5]=data4["totalPduDl"]
                data_oai[cnt,6]=data4["harqRound"]
                
                
                data5=data3["rrcMeasurements"]
                data_oai[cnt,7]=data5["pcellRsrp"]
                data_oai[cnt,8]=data5["pcellRsrq"]
                data6=data3["dlCqiReport"]["csiReport"][0]
                data_oai[cnt,9]=data6["p10csi"]["wbCqi"]
                # print(data6["p10csi"]["wbCqi"])
#             data[cnt,-1]=json_data['delayTargetBitrate']
#             data[cnt,-2]=json_data['rtt']
#             data[cnt,-3]=json_data['sendpkts']
#             data[cnt,-4]=json_data['averageLoss']
                cnt=cnt+1

            
    return(data_oai[:cnt,:])










# 用法示例
data_all_list=np.zeros((1,20))
for file_i in range(1,31):
    print(file_i)
    video_data1=read_json_files(f'../data/data_20240101/UE1_DATA_{file_i}_V/',1)
    # print("video_data1",video_data1)
    video_data1[1:,13:15]=video_data1[1:,13:15]-video_data1[:-1,13:15]
    video_data1=video_data1[1:,:]


    oai_data=read_json_files_oai(f'../data/data_20240101/1UE_data_{file_i}/',1)
    oai_data=oai_data[np.argsort(oai_data[:,0]),:]
    # print(oai_data[0,:2])
    oai_data1=oai_data[oai_data[:,1]==oai_data[0,1],:].copy()
    oai_data2=oai_data[oai_data[:,1]==oai_data[1,1],:].copy()


    oai_data1[1:,3:6]=oai_data1[1:,3:6]-oai_data1[:-1,3:6]
    oai_data1=oai_data1[1:,:]
    oai_data2[1:,3:6]=oai_data2[1:,3:6]-oai_data2[:-1,3:6]
    oai_data2=oai_data2[1:,:]

    oai_data=np.vstack((oai_data1,oai_data2))


    # data_all=video_data1.copy()
    # print(data_all.shape)
    # data_all=np.vstack((data_all,oai_data))
    # data_all=data_all[np.argsort(data_all[:,0]),:]
    # print(data_all)
    # import pandas as pd
    # data_save=pd.DataFrame(data_all)
    # data_save.to_csv(f"../data/data_20240101/UE1_{file_i}.csv")
    
    data1=oai_data.copy()
    data3=video_data1.copy()
    data1[:,-1]=np.round(data1[:,-0])
    data3[:,-1]=np.round(data3[:,-0])
    min_time1=min(data1[:,-1])
    max_time1=max(data1[:,-1])
    min_time3=min(data3[:,-1])
    max_time3=max(data3[:,-1])
    data_ue11=data1.copy()
    data_ue12=data3.copy()
    max_time=min(max_time1,max_time3)-15
    min_time=max(min_time1,min_time3)+20
    data_ue1=np.zeros((int(max_time-min_time),20))
    for time_i in range(0,int(max_time-min_time)):
        #0time/1rnti/2mcs/3sdubyte/4prb/5pdu/6harq/10UEidx/11bitrate/12rtt/13pkts/14payload/15loss
        ind1=np.where(data_ue11[:,-1]==time_i+min_time)[0]
        ind2=np.where(data_ue12[:,-1]==time_i+min_time)[0]
        data_ue1[time_i,0]=time_i+min_time
        data_ue1[time_i,1]=np.mean(data_ue11[ind1,1])
        data_ue1[time_i,2]=np.mean(data_ue11[ind1,2])
        data_ue1[time_i,3]=np.sum(data_ue11[ind1,3])
        data_ue1[time_i,4]=np.sum(data_ue11[ind1,4])
        data_ue1[time_i,5]=np.sum(data_ue11[ind1,5])
        data_ue1[time_i,6]=np.mean(data_ue11[ind1,6])
        data_ue1[time_i,7]=np.mean(data_ue11[ind1,7])
        data_ue1[time_i,8]=np.mean(data_ue11[ind1,8])
        data_ue1[time_i,9]=np.mean(data_ue11[ind1,9])
        data_ue1[time_i,10]=np.mean(data_ue12[ind2,10])
        data_ue1[time_i,11]=np.mean(data_ue12[ind2,11])
        data_ue1[time_i,12]=np.mean(data_ue12[ind2,12])
        data_ue1[time_i,13]=np.sum(data_ue12[ind2,13])
        data_ue1[time_i,14]=np.sum(data_ue12[ind2,14])
        data_ue1[time_i,15]=np.mean(data_ue12[ind2,15])
        
        data_save=pd.DataFrame(data_ue1[:,:])
        data_save.to_csv(f"../data/data_20240101/UE2_{file_i}_train.csv")
    
    data_all_list=np.vstack((data_all_list,data_ue1))
    print(len(data_ue1),len(data_all_list))
    
data_all_list=data_all_list[1:,:]
data_all_list=data_all_list[np.argsort(data_all_list[:,0]),:]
data_all_list[:,0]=np.array(range(len(data_all_list)))+data_all_list[0,0]
data_test=data_all_list.copy()
data_save=dict()
date=[]
mcs=[]
prb=[]
harq=[]
rsrp=[]
rsrq=[]
cqi=[]
OnRLQoE=[]
LokiQoE=[]
GeminiQoE=[]
# OnRLQoE=[]
#0time/1rnti/2mcs/3sdubyte/4prb/5pdu/6harq/7rsrp/8rsrq/9cqi/10UEidx/11bitrate/12rtt/13pkts/14payload/15loss
cnt=0
time_begin=data_test[0,0]
for i in range(1,len(data_test)):
    if data_test[i,12]==0:
        continue
    timestamp=time_begin+cnt
    cnt=cnt+1
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    formatted_date = dt_object.strftime("%m/%d/%Y %H:%M:%S")
    date.append(formatted_date)
    mcs.append(data_test[i,2])
    prb.append(data_test[i,4])
    harq.append(data_test[i,6])
    rsrp.append(data_test[i,7])
    rsrq.append(data_test[i,8])
    cqi.append(data_test[i,9])
    if (i>=1):
        qoe_onrl=50*data_test[i,11]/1e6-50*data_test[i,15]-10*data_test[i,12]-30*np.abs(data_test[i,11]-data_test[i-1,11])/1e6
    else: 
        qoe_onrl=50*data_test[i,11]/1e6-50*data_test[i,15]-10*data_test[i,12]
    OnRLQoE.append(qoe_onrl)
    if (i>=1):
        qoe_loki=5*(data_test[i,14]*8/1e6-data_test[i,15])/data_test[i,12]-0.5*np.abs(data_test[i,11]-data_test[i-1,11])/1e6
    else: 
        qoe_loki=5*(data_test[i,14]*8/1e6-data_test[i,15])/data_test[i,12]
    LokiQoE.append(qoe_loki)    
    if (i>=1):
        qoe_gemini=data_test[i,11]/1e3-1.5*data_test[i,15]-1.5*data_test[i,12]-0.02*np.abs(data_test[i,11]-data_test[i-1,11])/1e3
    else: 
        qoe_gemini=data_test[i,11]/1e3-1.5*data_test[i,15]-1.5*data_test[i,12]
    GeminiQoE.append(qoe_gemini)
print(cnt)
    
plt.plot(GeminiQoE)
plt.show()
plt.plot(LokiQoE)
plt.show()    
    
data_save["date"]=date
data_save["mcs"]=mcs
data_save["prb"]=prb
data_save["harq"]=harq
data_save["rsrp"]=rsrp
data_save["rsrq"]=rsrq
data_save["cqi"]=cqi
data_save["OnRLQoE"]=OnRLQoE
data_save["Loki_QoE"]=LokiQoE
data_save["GeminiQoE"]=GeminiQoE
df = pd.DataFrame(data_save)
df.to_csv('../QoEdataall.csv', index=False,header=["date","mcs","prb","harq","rsrp","rsrq","cqi","OnRLQoE","LokiQoE","GeminiQoE"])
# def max_min(data):
# #     print(min(data),max(data))
#     data=(data - min(data)) / (max(data) - min(data))
#     return data

# data_all2=data_all.copy()
# for i in range(2,20):
#     if i==10:
#         continue
#     data_all2[:,i]=max_min(data_all2[:,i])
# print(data_all)
# oai_data=data_all2[data_all2[:,1]>0,:]
# print(oai_data)
# oai_data1=oai_data[oai_data[:,1]==oai_data[0,1],:].copy()
# oai_data2=oai_data[oai_data[:,1]==oai_data[1,1],:].copy()   
# oai_video1=data_all2[data_all2[:,10]==10,:].copy() 
# oai_video2=data_all2[data_all2[:,10]==11,:].copy() 


# plt.plot(oai_data1[:,0],oai_data1[:,1])
# plt.plot(oai_data2[:,0],oai_data2[:,1])
# plt.plot(oai_video1[:,0],oai_video1[:,11])
# plt.plot(oai_video2[:,0],oai_video2[:,11])
# plt.show()
# plt.plot(oai_data1[:,0],oai_data1[:,2]-1)
# plt.plot(oai_data2[:,0],oai_data2[:,2])
# plt.plot(oai_video1[:,0],oai_video1[:,12]-1)
# plt.plot(oai_video2[:,0],oai_video2[:,12])
# plt.show()
# plt.plot(oai_data1[:,0],oai_data1[:,3]-1)
# plt.plot(oai_data2[:,0],oai_data2[:,3])
# plt.plot(oai_video1[:,0],oai_video1[:,13]-1)
# plt.plot(oai_video2[:,0],oai_video2[:,13])
# plt.show()
# plt.plot(oai_data1[:,0],oai_data1[:,4])
# plt.plot(oai_data2[:,0],oai_data2[:,4])
# plt.plot(oai_video1[:,0],oai_video1[:,14])
# plt.plot(oai_video2[:,0],oai_video2[:,14])
# plt.show()