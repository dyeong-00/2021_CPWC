import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import visdom

from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# LSTM block
class BlockLSTM(nn.Module):
    def __init__(self, time_steps, num_variables, lstm_hs=8, dropout=0., attention=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=time_steps, hidden_size=lstm_hs, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        # LSTM layer input shape: (batch_size, num_variables, time_steps)
        # Pass it through the LSTM layer
        x,_ = self.lstm(x)
        y = x[:,-1]
        
        #Dropout layer
        y = self.dropout(y)
        
        # output shape: (batch_size, hidden_size)
        return y       


    
# Conv1D block
class BlockFCNConv(nn.Module):
    def __init__(self, in_channel=1, out_channel=64, kernel_size=8,padding = 3, momentum=0.99, epsilon=0.001, squeeze=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size,padding=padding)
        nn.init.kaiming_uniform_(self.conv.weight)
        self.channel = out_channel
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Pass it through the conv1D layer
        x = self.conv(x)
        
        # ReLU activation function
        y = self.relu(x)
        
        # output shape: (batch_size, out_channel)
        return y

    


# FCN block
class BlockFCN(nn.Module):
    def __init__(self, time_steps, channels=[20, 64, 128, 64], kernels=[3, 4, 3],paddings=[0, 0, 0], mom=0.99, eps=0.001):
        super().__init__()
        self.conv1 = BlockFCNConv(channels[0], channels[1], kernels[0],paddings[0], momentum=mom, epsilon=eps, squeeze=True)
        self.conv2 = BlockFCNConv(channels[1], channels[2], kernels[1],paddings[1], momentum=mom, epsilon=eps, squeeze=True)
        self.conv3 = BlockFCNConv(channels[2], channels[3], kernels[2],paddings[2], momentum=mom, epsilon=eps)
        output_size = time_steps - sum(kernels) + len(kernels)
        self.global_pooling = nn.AvgPool1d(output_size)
        
    def forward(self, x):
        # FCN block input shape: (batch_size, num_variables, time_steps)
        # Pass it through three conv1D layer
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # apply Global Average Pooling 1D
        y = self.global_pooling(x)
        
        y = y.squeeze(dim=2)
        
        # output shape: (batch_size, out_channel)
        return y

    
    


# LSTM-FCN
class LSTMFCN(nn.Module):
    def __init__(self, time_steps, num_variables):
        super().__init__()
        self.lstm_block = BlockLSTM(time_steps, num_variables)
        self.fcn_block = BlockFCN(time_steps)
        self.FC = nn.Linear(8+64,3)
        
    def forward(self, x):
        # input shape: (batch_size, time_steps, num_variables)
        # Dimension shuffle layer
        x = torch.transpose(x,2,1)
        
        # pass input through LSTM block
        x1 = self.lstm_block(x)
        
        # pass input through FCN block
        x2 = self.fcn_block(x)
        
        # concatenate two outputs
        x = torch.cat([x1, x2], 1)
        
        # pass through Softmax activation function
        y = self.FC(x)
        
        return y    
   
  