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
learning_rate=1e-4
epochs=200

train_dataset = MyDataset(data="train_inter")
test_dataset = MyDataset(data="train")
inter_train_dataset = Subset(train_dataset, range(0, 500000))
inter_test_dataset = Subset(train_dataset, range(500000, 750000))
inter_train_loader = DataLoader(inter_train_dataset, batch_size = batch_size, drop_last=True, shuffle=True)
inter_test_loader = DataLoader(inter_test_dataset, batch_size = batch_size, drop_last=True, shuffle=False)

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
# optimizer = optim.SGD(net.parameters(), lr=learning_rate)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# criteon = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8])).to(device)
criteon = nn.CrossEntropyLoss().to(device)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                           milestones=[100, 150], gamma=0.1)
# criteon = nn.CrossEntropyLoss()
# a=torch.randn(79,38)
# t=net(a)
# print(t)
for epoch in range(epochs):
    net.train()
    batch_idx=0
    total_one = 0
    for (data, target) in inter_train_loader:
        data, target = data.to(device), target.to(device)
        # data, target = data, target
        logits = net(data.to(torch.float32))
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(), w2.grad.norm())
        optimizer.step()
        batch_idx+=1
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(inter_train_dataset),
                       100. * batch_idx * len(data) / len(inter_train_dataset), loss.item()))
        total_one += target.sum()
    print("train total one: ", total_one)
    scheduler.step()
    with torch.no_grad():
        net.eval()
        test_loss = 0
        correct = 0
        total_one = 0
        for data, target in inter_test_loader:
            data, target = data.to(device), target.cuda()
            # data, target = data, target
            logits = net(data.to(torch.float32))
            test_loss += criteon(logits, target).item()
            pred = logits.data.max(1)[1]
            correct += pred.eq(target.data).sum()
            total_one += target.sum()
        print("total one: ", total_one)
        test_loss /= len(test_dataset)
        print('\nInter test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(inter_test_dataset),
            100. * correct / len(inter_test_dataset)))

    with torch.no_grad():
        net.eval()
        test_loss = 0
        correct = 0
        total_one = 0
        for data, target in test_loader:
            data, target = data.to(device), target.cuda()
            # data, target = data, target
            logits = net(data.to(torch.float32))
            test_loss += criteon(logits, target).item()
            pred = logits.data.max(1)[1]
            correct += pred.eq(target.data).sum()
            total_one += target.sum()
        print("total one: ", total_one)
        test_loss /= len(test_dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_dataset),
            100. * correct / len(test_dataset)))