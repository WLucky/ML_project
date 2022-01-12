import pdb

import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
from    torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from dataset import MyDataset
from sampler import AverageSampler

batch_size=200
learning_rate=0.0001
epochs=20

train_dataset = MyDataset(data="train_inter")
test_dataset = MyDataset(data="train")
# inter_train_dataset = Subset(train_dataset, range(0, 500000))
inter_test_dataset = Subset(train_dataset, range(500000, 750000))
inter_train_dataset = train_dataset
# inter_train_loader = DataLoader(inter_train_dataset, batch_size = batch_size, drop_last=True, shuffle=True)
inter_test_loader = DataLoader(inter_test_dataset, batch_size = batch_size, drop_last=True, shuffle=False)
train_sampler = AverageSampler(train_dataset, batch_size)
inter_train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
# train_loader=DataLoader(train_dataset, batch_size = batch_size, drop_last=True, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(38, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
            nn.Softmax()
        )

        self.__init_parameters()    

    def __init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.model(x)

        return x

device = torch.device('cuda:0')
net = MLP().to(device)
# net = MLP()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
# criteon = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8])).to(device)
criteon = nn.CrossEntropyLoss().to(device)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                           milestones=[10], gamma=0.1)
# criteon = nn.CrossEntropyLoss()
# a=torch.randn(79,38)
# t=net(a)
# print(t)
total_iter = 10000
cur_iter = 0
test_iter = 300

for (data, target) in inter_train_loader:
    data, target = data.to(device), target.to(device)
    # data, target = data, target
    logits = net(data.to(torch.float32))
    loss = criteon(logits, target)

    optimizer.zero_grad()
    loss.backward()
    # print(w1.grad.norm(), w2.grad.norm())
    optimizer.step()
    # scheduler.step()
    cur_iter+=1
    if cur_iter % 50 == 0:
        print('Train iter:[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            cur_iter, total_iter,
                   100. * cur_iter / total_iter, loss.item()))

    if cur_iter % test_iter == 0 and cur_iter != 0:
        with torch.no_grad():
            net.eval()
            test_loss = 0
            correct = 0
            total_one = 0
            cur_test_iter = 0
            for data, target in inter_train_loader:
                data, target = data.to(device), target.cuda()
                # data, target = data, target
                logits = net(data.to(torch.float32))
                test_loss += criteon(logits, target).item()
                pred = logits.data.max(1)[1]
                correct += pred.eq(target.data).sum()
                total_one += target.sum()
                cur_test_iter+=1
                if cur_test_iter >= 99:
                    break
            print("total one: ", total_one)
            test_loss /= batch_size * 100
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(inter_test_dataset),
                100. * correct / (batch_size * 100)))

        if cur_iter >= total_iter:
            break