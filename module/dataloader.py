import torch
import numpy as np
from torchvision.datasets import CIFAR10

def data_tf(x):
	x=np.array(x,	dtype='float32')	/ 255
	x=(x - 0.5)	/ 0.5 #	标准化
	x=x.transpose((2,0,1))	#将channel放在第一维
	x=torch.from_numpy(x)
	return x
				
train_set=CIFAR10('./data',	train=True,transform=data_tf)
train_data=torch.utils.data.DataLoader(train_set,batch_size=64,shuffle=True)
test_set=CIFAR10('./data',	train=False,transform=data_tf)
test_data=torch.utils.data.DataLoader(test_set,batch_size=128,shuffle=False)
