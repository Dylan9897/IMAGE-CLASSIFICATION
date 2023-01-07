import torch
from torch import nn
from torch.autograd import Variable
import argparse
from datetime import datetime
import torchsummary

from module.config import parserConfig
from models.model import Model
from models.Ranger import opt_func
from module.dataloader import train_data,test_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Image Classification')
parser.add_argument('--model', type=str, default="AlexNet", help='choose a model')
args = parser.parse_args()

config = parserConfig(args)
if args.model == "GoogleNet":
    from models.GoogleNet import Model
model = Model(config)
print(model)
s = input()

# 定义优化器
optimizer = opt_func(model.parameters())
# optimizer=torch.optim.SGD(model.parameters(),	lr=1e-1)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().data
    return num_correct / total


def train(net, train_data, valid_data, num_epochs, optimizer, criterion):
    if torch.cuda.is_available():
        net = net.cuda()
    prev_time = datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for im, label in train_data:
            print(im.shape)
            s = input()
            if torch.cuda.is_available():
                im = Variable(im.cuda())  # (bs, 3, h, w)
                label = Variable(label.cuda())  # (bs, h, w)
            else:
                im = Variable(im)
                label = Variable(label)
            # forward
            output = net(im)
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data
            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    im = Variable(im.cuda(), volatile=True)
                    label = Variable(label.cuda(), volatile=True)
                else:
                    im = Variable(im, volatile=True)
                    label = Variable(label, volatile=True)
                output = net(im)
                loss = criterion(output, label)
                valid_loss += loss.data
                valid_acc += get_acc(output, label)
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)



if __name__=="__main__":
    # input_demo=Variable(torch.zeros(1,3,32,32)).to(DEVICE)
    
    # torchsummary.summary(tmodel,(1,32,100))
    # torchsummary.summary(model,input_demo)
    # output_demo = model(input_demo)
    # print(output_demo.shape)
    train(model,train_data,test_data,20,optimizer,criterion)

