import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import visdom
import socket

from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from LSTM_FCN import *

time_step=30

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
    
print('Using Pytorch Version: ',torch.__version__,'Device: ',DEVICE)

#모델 불러오기
model=LSTMFCN(30, 20) #model 선언하기
#미리 train하고 저장한 state_dict 불러오기
model.load_state_dict(torch.load('model/train_0927.pth'))
model = model.to(DEVICE) #model GPU에 불러오기
model.eval() #model 평가모드로 전환하기

def load_data(inputdata):
    #소켓에서 들어온 데이터 전처리
    data = client_socket_1.recv(1024)
    logdata = repr(data.decode('utf-8'))
    
    logdata = str(logdata)
    logdata = np.fromstring(logdata.strip("''"), dtype=float, sep=',') #string 나눠서 np로 변환
    #print("log:",logdata)
    logdata = np.reshape(logdata, ((1,)+logdata.shape))
    inputdata = np.vstack([inputdata, logdata]) #inputdata에 logdata 추가
    
    return inputdata

HOST = '210.123.37.216'  
# 서버에서 지정해 놓은 포트 번호입니다. 
PORT_1 = 1000 #LogDisplay
    

# 소켓 객체를 생성합니다. 
# 주소 체계(address family)로 IPv4, 소켓 타입으로 TCP 사용합니다.  
client_socket_1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


# 지정한 HOST와 PORT를 사용하여 서버에 접속합니다. 
client_socket_1.connect((HOST, PORT_1))


inputdata = np.zeros((1, 22)) #inputdata 초기화
count = 0
while True: #29 frame
    inputdata = load_data(inputdata)
    if len(inputdata) == 30:
        break
    
inputdata = np.delete(inputdata, 0, 0) #초기화할 때 사용한 행 삭제

while True:
    inputdata = load_data(inputdata)
    
    #inputdata 전처리하기
    test_x = np.reshape(inputdata, ((1,) + inputdata.shape))
    #print(test_x.shape)
    test_x = torch.tensor(test_x[:, :, 2:], dtype=torch.float32)
    test_x = test_x.to(DEVICE)
    
    #model에서 output받기
    output = model(test_x)
    prediction = output.max(1, keepdim = True)[1]
    
    prediction=prediction[0].cpu().numpy()
    print("log: ", inputdata[-1])
    print("meanuver: ",prediction[0])
    #inputdata의 마지막 행과 output 합치기
    pre=np.hstack((inputdata[-1],prediction))
    inputdata = np.delete(inputdata, 0, 0) #다음 loop를 위해 첫행 삭제하기
    client_socket_1.send(str(prediction[0]).encode())