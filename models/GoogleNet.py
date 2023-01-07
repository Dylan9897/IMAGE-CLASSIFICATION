import torch
from torch import nn
import numpy as np
import torchsummary
# from models.Mish import Mish

EPS = 1e-3

# 定义一个基本的层结构
def conn_relu(in_channel,out_channel,kernel,stride=1,padding=0)->nn.Module:
    layer = nn.Sequential(
        nn.Conv2d(in_channel,out_channel,kernel,stride,padding),
        nn.BatchNorm2d(out_channel,eps=EPS),
        nn.ReLU()
    )
    return layer

class Inception(nn.Module):
    def __init__(self,in_channel,out1_1,out2_1,out2_3,out3_1,out3_5,out4_1):
        super(Inception,self).__init__()

        # 第一条路线
        self.branch1_1 = conn_relu(in_channel,out1_1,1)

        # 第二条路线
        self.branch3_3 = nn.Sequential(
            conn_relu(in_channel,out2_1,1),
            conn_relu(out2_1,out2_3,3,padding=1)
        ) 

        # 第三条路线
        self.branch5_5 = nn.Sequential(
            conn_relu(in_channel,out3_1,1),
            conn_relu(out3_1,out3_5,5,padding=2)
        )

        # 第四条路线
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3,stride=1,padding=1),
            conn_relu(in_channel,out4_1,1)
        )

    def forward(self,x):
        f1 = self.branch1_1(x)
        f2 = self.branch3_3(x)
        f3 = self.branch5_5(x)
        f4 = self.branch_pool(x)
        output = torch.cat((f1,f2,f3,f4),dim=1)
        return output

class Model(nn.Module):
    def __init__(self,archs,verbose=False):
        super(Model,self).__init__()
        self.verbose = verbose
        self.archs = archs
        print("asdasdad")
        self.blocks = self.get_blocks()
        print(self.blocks)

    
    def get_blocks(self):
        blocks = {}
        for i,archs in enumerate(self.archs):
            temp = []
            for j in range(1,len(archs)):
                arch = archs[j]
                if arch[0] == "conn_relu":
                    temp.append(conn_relu(arch[1],arch[2],arch[3],arch[4],arch[5]))
                elif arch[0] == "Maxpool":
                    temp.append(nn.MaxPool2d(arch[1],arch[2],arch[3]))
                elif arch[0] == "inception":
                    temp.append(Inception(arch[1],arch[2],arch[3],arch[4],arch[5],arch[6]))
                elif arch[0] == "Avgpool":
                    temp.append(nn.AvgPool2d(arch[1],arch[2],arch[3]))
                elif arch[0] == "Linear":
                    temp.append(nn.Linear(arch[1], arch[2]))
            blocks[archs[0]] = nn.Sequential(*temp)
        return blocks
        
    def forward(self,x):
        print("asdadsadadasd")
        # keywords = ["block1","block2","block3","block4","block5","classify"]
        x = self.blocks["block1"](x)
        if self.verbose:
            print("block 1 output:{}".format(x.shape))
        return x







if __name__=="__main__":
    from torch.autograd import Variable
    test_net = Inception(3,64,48,64,64,96,32)
    test_x = Variable(torch.zeros(1,3,96,96))
    print("input shape {} * {} *{}".format(test_x.shape[1],test_x.shape[2],test_x.shape[3]))
    test_y = test_net(test_x)
    print("output shape:{} * {} * {}".format(test_y.shape[1],test_y.shape[2],test_y.shape[3]))


