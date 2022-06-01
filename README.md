# 2021_CPWC Maneuver Prediction Model

차선 변경 가능 여부 판단 딥러닝 모델을 구축하였다.

## Model Introduction
LSTM layer와 1D Convolution layer 3개를 합친 LSTM-FCN 모델을 사용하였으며 Pytorch로 구현하였다.   
LSTM-FCN은 시계열 데이터 분류 작업에서 단순 LSTM보다 더 좋은 성능을 달성하는 것으로 알려진 모델이다.    
모델 학습 전에 MinMaxScaler를 이용하여 최대 최소 정규화를 진행하였다. 또한 LSTM 블록에서 Dimension Shuffle layer를 통해 train 시간을 단축하였으며 overfitting을 방지하였다.    
input data는 10Hz로 받았으며 time step은 30이다.   
최종 output은 0 (lane keeping), 1 (left lane change), 2 (right lane change)로 세가지이며 이는 input feature와 합쳐져서 Trajectory Prediction 모델로 넘어간다.    

## Model Input & Output
**Input Feature**   
Ego: Speed, Acceleration, Yaw angle   
현재 차선 정보(끝차선에 있는지): Left edge, Right edge   
LB(Left Behind car): Position x, Position y, Speed, Acceleration   
LF(Left Front car): Position x, Position y, Speed, Acceleration   
B(Behind car): Position x, Position y, Speed, Acceleration   
F(Front car): Position x, Position y, Speed, Acceleration   
RB(Rignt Behind car): Position x, Position y, Speed, Acceleration   
RF(Rignt Front car): Position x, Position y, Speed, Acceleration   
 

**Output**   
0: lane keeping   
1: left lane change   
2: right lane change   

## How to Train Behavior Model
Open the ubuntu terminal and go to the path of the folder where behavior_train.py is located.
```
$ cd path/to/folder
```
Enter the following command to train the model
```
$ python behavior_train.py
```
Enter the path of the csv file and batch size, learning rate for train in the terminal.   
You can check the progress at the terminal

## Model Result
<img src="/result.png" width="500px" title="result" alt="result"></img><br/>
