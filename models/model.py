import torch
import torchsummary
from models.Mish import Mish

class Conv_Block(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super(Conv_Block,self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,stride,padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        # self.act = torch.nn.ReLU()
        self.act = Mish()
    
    def forward(self,x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        return out
    
class Linear_Block(torch.nn.Module):
    def __init__(self,infeatures,outfeatures,tag=True):
        super(Linear_Block,self).__init__()
        self.linear = torch.nn.Linear(infeatures, outfeatures)
        self.tag = tag
        if self.tag:
            # self.act = torch.nn.ReLU()
            self.act = Mish()
    
    def forward(self,x):
        x = x.view(x.shape[0],-1)
        out = self.linear(x)
        if self.tag:
            out = self.act(out)
        return out

class Reshape_Block(torch.nn.Module):
    def __init__(self,shape):
        super(Reshape_Block,self).__init__()
        self.shape = shape
    
    def forward(self,x):
        out = x.reshape(self.shape)
        return out


class Model_NN(torch.nn.Module):
    def __init__(self,arch,model_name=None):
        super(Model_NN,self).__init__()
        self.arch = arch
        self.model = self.getModel()
    
    def getModel(self):
        layers=[]
        for L in self.arch:
            if L[0]=="Conv2D":
                layers.append(Conv_Block(L[1],L[2],L[3],L[4],L[5]))
            elif L[0]=="Maxpool":
                layers.append(torch.nn.MaxPool2d(L[1],L[2],L[3]))
            elif L[0]=="Reshape":
                layers.append(Reshape_Block(L[1]))
            elif L[0]=="Conn":
                layers.append(Linear_Block(L[1],L[2]))
            elif L[0]=="LastLinear":
                layers.append(Linear_Block(L[1],L[2],tag=False))
        return torch.nn.Sequential(*layers)
    
    def forward(self,x):
        out = self.model(x)
        return out

def Model(config,device):
    model = Model_NN(config["params"],config["model"]).to(device)
    return model
