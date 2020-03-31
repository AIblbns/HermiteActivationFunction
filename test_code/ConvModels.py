import numpy as np
from numpy.polynomial.hermite import Hermite, hermval
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class sigmoidNet(nn.Module):
        def __init__(self,channels,hidden_size):
            super(sigmoidNet, self).__init__()
            self.conv1 = nn.Conv2d(channels, 16, kernel_size=5)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(hidden_size, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear (84, 10)
      
        def forward(self, x):
            x = self.pool1(torch.sigmoid(self.conv1(x)))
            x = self.pool2(torch.sigmoid(self.conv2(x)))
            x = x.view(-1, x.shape[1]* x.shape[2]* x.shape[3])
            x = torch.sigmoid(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))
            out=self.fc3(x)
            return out


class tanhNet(nn.Module):
        def __init__(self,channels,hidden_size):
            super(tanhNet, self).__init__()
            self.conv1 = nn.Conv2d(channels, 16, kernel_size=5)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(hidden_size, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear (84, 10)
      
        def forward(self, x):
            x = self.pool1(torch.tanh(self.conv1(x)))
            x = self.pool2(torch.tanh(self.conv2(x)))
            x = x.view(-1, x.shape[1]* x.shape[2]* x.shape[3])
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            out=self.fc3(x)
            return out


class ReluNet(nn.Module):
    def __init__(self,channels,hidden_size):
        super(ReluNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)            
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(hidden_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear (84, 10)
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1,  x.shape[1]* x.shape[2]* x.shape[3])
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out=self.fc3(x)
        return out
    

class LeakyReluNet(nn.Module):
    def __init__(self,channels,hidden_size):
        super(LeakyReluNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(hidden_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear (84, 10)
    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.pool2(F.leaky_relu(self.conv2(x)))
        x = x.view(-1,  x.shape[1]* x.shape[2]* x.shape[3])
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        out=self.fc3(x)
        return out


class ELUNet(nn.Module):
    def __init__(self,channels,hidden_size):
        super(ELUNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(hidden_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear (84, 10)
    def forward(self, x):
        x = self.pool1(F.elu(self.conv1(x)))
        x = self.pool2(F.elu(self.conv2(x)))
        x = x.view(-1,  x.shape[1]* x.shape[2]* x.shape[3])
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        out=self.fc3(x)
        return out


class SeluNet(nn.Module):
    def __init__(self,channels,hidden_size):
        super(SeluNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(hidden_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear (84, 10)
    def forward(self, x):
        x = self.pool1(F.selu(self.conv1(x)))
        x = self.pool2(F.selu(self.conv2(x)))
        x = x.view(-1,  x.shape[1]* x.shape[2]* x.shape[3])
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        out=self.fc3(x)
        return out


class SwishNN(nn.Module):
    def __init__(self,channels,hidden_size):
        super(SwishNN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(hidden_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear (84, 10)
    def forward(self, x):
        x = self.pool1(Swishactivation(self.conv1(x)))
        x = self.pool2(Swishactivation(self.conv2(x)))
        x = x.view(-1, x.shape[1]* x.shape[2]* x.shape[3])
        x = Swishactivation(self.fc1(x))
        x = Swishactivation(self.fc2(x))
        out=self.fc3(x)
        return out
def Swishactivation(x):
    x=x*torch.sigmoid(x)
    return x
    

class MishNet(nn.Module):
    def __init__(self,channels,hidden_size):
        super(MishNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(hidden_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear (84, 10)
    def forward(self, x):
        x = self.pool1(Mishactivation(self.conv1(x)))
        x = self.pool2(Mishactivation(self.conv2(x)))
        x = x.view(-1,  x.shape[1]* x.shape[2]* x.shape[3])
        x = Mishactivation(self.fc1(x))
        x = Mishactivation(self.fc2(x))
        out=self.fc3(x)
        return out
def Mishactivation(x):
    x=x * torch.tanh(F.softplus(x))
    return x
    

class ConvNet01(nn.Module):
    def __init__(self,channels,hidden_size):
        super(ConvNet01, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)        
        self.fc1 = nn.Linear(3136, 10)

    def forward(self, x,coef):
        x=self.conv1(x)
        x= x.detach().numpy()
        x=hermval(x,coef,tensor=True)
        x = self.pool1(torch.from_numpy(x))
        x = x.view(-1, x.shape[1]* x.shape[2]* x.shape[3])
        x = x.float()
        out=self.fc1(x)
        return out  
 
    
class ConvNet1(nn.Module):
    def __init__(self,channels,hidden_size):
        super(ConvNet1, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)        
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(hidden_size, 10)

    def forward(self, x,coef):
        x=self.conv1(x)
        x=x.detach().numpy()
        x=hermval(x,coef,tensor=True)
        x=self.pool1(torch.from_numpy(x))
        x=x.float()
        x=Variable(x)
        x=self.conv2(x)
        x=x.detach().numpy()
        x=hermval(x,coef,tensor=True)
        x = self.pool2(torch.from_numpy(x))
        x = x.view(-1, x.shape[1]* x.shape[2]* x.shape[3])
        x = x.float()
        out=self.fc1(x)
        return out  


class ConvNet2(nn.Module):
    def __init__(self,channels,hidden_size):
        super(ConvNet2, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)        
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)   
        self.fc1 = nn.Linear(hidden_size, 100)
        self.fc2 = nn.Linear (100, 10)
    def forward(self, x,coef):
        x=self.conv1(x)
        x= x.detach().numpy()
        x=hermval(x,coef,tensor=True)
        x = self.pool1(torch.from_numpy(x))
        x = x.float()
        x=self.conv2(x)
        x= x.detach().numpy()
        x=hermval(x,coef,tensor=True)
        x = self.pool2(torch.from_numpy(x))  
        x = x.view(-1, x.shape[1]* x.shape[2]* x.shape[3])
        x = hermval(self.fc1(x),coef,tensor=True)
        x = x.float()
        out=self.fc2(x)
        return out

    
class ConvNet3(nn.Module):
    def __init__(self,channels,hidden_size):
        super(ConvNet3, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)        
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(hidden_size, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear (100, 10)
    def forward(self, x,coef):
        x=self.conv1(x)
        x= x.detach().numpy()
        x=hermval(x,coef,tensor=True)
        x = self.pool1(torch.from_numpy(x))
        x = x.float()
        x=self.conv2(x)
        x= x.detach().numpy()
        x=hermval(x,coef,tensor=True)
        x = self.pool2(torch.from_numpy(x))
        x = x.view(-1, x.shape[1]* x.shape[2]* x.shape[3])
        x = hermval(self.fc1(x),coef,tensor=True)
        x = hermval(self.fc2(x),coef,tensor=True)
        x = x.float()
        out=self.fc3(x)
        return out


class ConvNet4(nn.Module):
    def __init__(self,channels,hidden_size):
        super(ConvNet4, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)        
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(hidden_size, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear (50, 10)
    def forward(self, x,coef):
        x=self.conv1(x)
        x= x.detach().numpy()
        x=hermval(x,coef,tensor=True)
        x = self.pool1(torch.from_numpy(x))
        x = x.float()
        x = self.conv2_bn(x)        
        x= x.detach().numpy()
        x=hermval(x,coef,tensor=True)
        x = self.pool2(torch.from_numpy(x))
        x = x.view(-1, x.shape[1]* x.shape[2]* x.shape[3])
        x = hermval(self.fc1(x),coef,tensor=True)
        x = hermval(self.fc2(x),coef,tensor=True)
        x = hermval(self.fc3(x),coef,tensor=True)
        x = x.float()
        out=self.fc4(x)
        return out


class DynamicNet(nn.Module):
    def __init__(self,channels,hidden_size):
        super(DynamicNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)        
        self.fc1 = nn.Linear(hidden_size, 10)

    def forward(self, x,coef):
        x=self.conv1(x)
        x= x.detach().numpy()
        x=hermval(x,coef,tensor=True)
        x = self.pool1(torch.from_numpy(x))
        x = x.view(-1, x.shape[1]* x.shape[2]* x.shape[3])
        x = x.float()
        out=self.fc1(x)
        return out 
    

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(ResidualBlock)
  