import torch
from torch import nn
from torch.nn import MaxPool1d, Flatten, BatchNorm1d, LazyLinear, Conv1d, Dropout
from torch.nn import ReLU, Softmax
import torch.nn.functional as F

class base_model(nn.Module):
    def __init__(self):
        super(base_model, self).__init__()
        self.conv1 = Conv1d(in_channels=2, out_channels=64, kernel_size=7, padding=3)
        self.mp1 = MaxPool1d(2)

        self.conv2 = Conv1d(in_channels=64, out_channels=64, kernel_size=7, padding=3)
        self.mp2 = MaxPool1d(2)

        self.conv3 = Conv1d(in_channels=64, out_channels=64, kernel_size=7, padding=3)
        self.mp3 = MaxPool1d(2)

        self.conv4 = Conv1d(in_channels=64, out_channels=64, kernel_size=7, padding=3)
        self.mp4 = MaxPool1d(2)

        self.conv5 = Conv1d(in_channels=64, out_channels=64, kernel_size=7, padding=3)
        self.mp5 = MaxPool1d(2)

        self.conv6 = Conv1d(in_channels=64, out_channels=64, kernel_size=7, padding=3)
        self.mp6 = MaxPool1d(2)

        self.conv7 = Conv1d(in_channels=64, out_channels=64, kernel_size=7, padding=3)
        self.mp7 = MaxPool1d(2)

        self.flatten = Flatten()

        self.linear1 = LazyLinear(128)
        self.linear2 = LazyLinear(128)
        self.linear3 = LazyLinear(12)


    def forward(self,x):
        x = self.conv1(x)
        x = self.mp1(x)
        x = F.relu(x)        

        x = self.conv2(x)
        x = self.mp2(x)
        x = F.relu(x)  

        x = self.conv3(x)
        x = self.mp3(x)
        x = F.relu(x) 

        x = self.conv4(x)
        x = self.mp4(x)
        x = F.relu(x) 

        x = self.conv5(x)
        x = self.mp5(x)
        x = F.relu(x)  

        x = self.conv6(x)
        x = self.mp6(x)
        x = F.relu(x)   

        x = self.conv7(x)
        x = self.mp7(x)
        x = F.relu(x)  

        x = self.flatten(x)

        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = F.relu(x)
    
        output = self.linear3(x)
        return output

