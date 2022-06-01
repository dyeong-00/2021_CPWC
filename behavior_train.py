import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import visdom

from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from LSTM_FCN import *
#from LSTM import *

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')
print('Using Pytorch Version: ',torch.__version__,'Device: ',DEVICE)

vis = visdom.Visdom(env='train_maneuver') #Visualizing model results.



#Import data from the csv file and create a dataset.
class FeatureDataset(Dataset):
    def __init__(self, path,time_step):
        _x, _y = [], []
        file_out = pd.read_csv(path)
        
        #Separate input and output.
        x = file_out.iloc[:,0:-1].values
        y = file_out.iloc[:,-1].values

        #Create a window according to time_step
        for i in range(time_step, len(y)):
            _x.append(x[i-time_step:i,:]) 
            _y.append(y[i-1])

        self.x = np.array(_x)
        self.y = np.array(_y)
        print(self.x.shape)
        print(sum(self.y==0))
        print(sum(self.y==1))
        print(sum(self.y==2))
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]
        
    
    
def train(model, train_loader, optimizer):
        cnt_0 = 0
        cnt_1 = 0
        cnt_2 = 0
        train_loss = []
        train_correct = 0
        cnt = len(train_loader)
        
        # model train mode
        model.train()
        
        for train_x, train_y in train_loader:
            # Send train_x, train_y to DEVICE.
            train_x = train_x.to(DEVICE)
            train_y = train_y.to(DEVICE)
            
            # Initialize the gradient
            optimizer.zero_grad()
            
            # Put input into the model and get output.
            output = model(train_x)
            
            # Compare output with train_y to calculate loss.
            # We use Cross Entropy Loss
            weights = [0.2, 0.8, 0.8]
            class_weights = torch.FloatTensor(weights).to(DEVICE)
            CELoss =  nn.CrossEntropyLoss(weight=class_weights)
            loss = CELoss(output, train_y)
            
            # Calculate loss with a backpropagation algorithm.
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Find the result from output.
            prediction = output.max(1, keepdim = True)[1]
            for i in range(len(prediction)):
                pre = int(prediction[i])
                if pre == 0:
                    cnt_0 +=1
                elif pre == 1:
                    cnt_1 +=1
                elif pre == 2:
                    cnt_2 +=1
            
            #Calculate train accuracy and loss
            train_correct += prediction.eq(train_y.view_as(prediction)).sum().item()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)
        train_accuracy = 100. * train_correct / (len(train_loader)*BATCH_SIZE)
        
        # Visualize train loss and accuracy
        if epoch % 2 == 0:
            print("Epoch: {} \n".format(epoch))
            print("Train Loss: {:.4f}, Train Accuracy: {:.2f} % \n".format(train_loss, train_accuracy))
            
        return train_loss, train_accuracy

    
    
def evaluate(model, test_loader):
        cnt_0 = 0
        cnt_1 = 0
        cnt_2 = 0
        test_loss = []
        test_correct = 0
        cnt = len(test_loader)
        
        model.eval()

        with torch.no_grad():
            for test_x, test_y in test_loader:
                # Send test_x, test_y to DEVICE.
                test_x = test_x.to(DEVICE)
                test_y = test_y.to(DEVICE)
                
                # Put input into the model and get output.
                output = model(test_x)
                
                # Compare output with train_y to calculate loss.
                # We use Cross Entropy Loss
                weights = [0.2, 0.8, 0.8]
                class_weights = torch.FloatTensor(weights).to(DEVICE)
                CELoss =  nn.CrossEntropyLoss(weight=class_weights)
                loss =  CELoss(output, test_y)
                
                # Find the result from output.
                prediction = output.max(1, keepdim = True)[1]
                for i in range(len(prediction)):
                    pre = int(prediction[i])
                    if pre == 0:
                        cnt_0 +=1
                    elif pre == 1:
                        cnt_1 +=1
                    elif pre == 2:
                        cnt_2 +=1

                #Calculate test accuracy and loss
                test_correct += prediction.eq(test_y.view_as(prediction)).sum().item()
                test_loss.append(loss.item())
            test_loss = np.mean(test_loss)
            test_accuracy = 100. * test_correct / (len(test_loader)*BATCH_SIZE)
    
            # Visualize test loss and accuracy
            if epoch % 2 == 0:
                print("Test Loss: {:.4f}, Test Accuracy: {:.2f} % \n".format(test_loss, test_accuracy))
                print(cnt_0, '  ', cnt_1, '  ', cnt_2)
                
        return test_loss, test_accuracy

if __name__ == "__main__":
    # You can enter the path of the csv file for train and test in the terminal
    #data = input("Enter the path of the csv file for train and test: ")
    data = "data/train_5.csv"
    # load dataset from csv file
    Dataset = FeatureDataset(data,30) 

    # Split the dataset for train and test
    x_train, x_test, y_train, y_test = train_test_split(Dataset.x,Dataset.y,test_size =0.3,shuffle=False)
    y_test_label = y_test
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_test= torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Make Tensor dataset
    train_dataset = TensorDataset(x_train , y_train)
    test_dataset = TensorDataset(x_test , y_test)

    # Recommend 1000 epochs
    EPOCHS = 1000

    # You can enter the batch size and learning rate for model
    #BATCH_SIZE = int(input("Enter the batch size(recommend 128): "))
    #learning_rate = float(input("Enter the learning rate(recommend 0.0001): "))
    BATCH_SIZE = 128
    learning_rate = 0.0001
    val_losses = []
    loss_min = np.inf

    # Define behavior model, optimizer, scheduler
    model = LSTMFCN(30,20).to(DEVICE)
    #model = LSTM(30,20).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60, 80], gamma= 0.5)

    # Define train loader and test loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)

    # Code for visdom
    tit = '_' + str(BATCH_SIZE) +'_' + str(learning_rate)
    loss_plt = vis.line(Y = torch.Tensor(1,2).zero_(),
                        opts = dict(title = 'loss'+tit, legend = ['train_loss','test_loss'],
                                    showlegend=True))  
    accuracy_plt = vis.line(Y = torch.Tensor(1,2).zero_(),
                        opts = dict(title = 'accuracy'+tit, legend = ['train_accuracy','test_accuracy'],
    showlegend=True))

    print("BATCH_SIZE : ",BATCH_SIZE, "lr = ",learning_rate)

    for epoch in range(EPOCHS):
        if epoch % 2 == 0:
            print('--------------------------------------\n')
        train_loss, train_accuracy = train(model, train_loader, optimizer)

        # Save model
        val_losses.append(train_loss)
        if np.mean(val_losses) < loss_min:
            torch.save(model.state_dict(),'model/train_{}_{}.pth'.format(BATCH_SIZE, learning_rate))
            loss_min = np.mean(val_losses)

        loss, accuracy = evaluate(model, test_loader)
        loss = torch.Tensor([[train_loss,loss]])
        accuracy = torch.Tensor([[train_accuracy,accuracy]])
        vis.line(X = torch.Tensor([epoch]), Y = loss, win=loss_plt,update = 'append')
        vis.line(X = torch.Tensor([epoch]), Y = accuracy, win=accuracy_plt,update = 'append')
