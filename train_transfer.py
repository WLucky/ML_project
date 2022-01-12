import pdb
from pickle import FALSE, TRUE

import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
from    torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from dataset import MyDataset
from sampler import AverageSampler
import sys
from utils import *
import os.path as osp
from torch.utils.data import ConcatDataset

# stages = ["pretrain", "transfer"]
stages = ["transfer"]
save_path = "result9"
pretrain_path = "result1"
mkdir(save_path)

log_pth = osp.join(save_path, "log1.txt")
file = open(log_pth, 'w+')
sys.stdout = file
print("This message is for file!")

train_batch_size=7500
test_batch_size = 2500
learning_rate=0.000001
epochs=200

print("lr: ", learning_rate)


inter_train_dataset = MyDataset(data="train_inter")
train_dataset = MyDataset(data="train")
####### inter data #####
inter_test_dataset = Subset(inter_train_dataset, range(500000, 750000))
inter_test_loader = DataLoader(inter_test_dataset, batch_size = train_batch_size, drop_last=True, shuffle=False)
train_sampler = AverageSampler(inter_train_dataset, train_batch_size)
inter_train_loader = DataLoader(inter_train_dataset, batch_sampler=train_sampler)

####### data ###########
sub_train_dataset = Subset(train_dataset, range(0, 7500))
sub_test_dataset = Subset(train_dataset, range(7500, 10000))

concat_dataset = ConcatDataset([inter_train_dataset, sub_train_dataset])

train_loader=DataLoader(sub_train_dataset, batch_size = train_batch_size, drop_last=True, shuffle=True)
train2_loader=DataLoader(concat_dataset , batch_size = train_batch_size, drop_last=True, shuffle=True)

test_loader=DataLoader(sub_test_dataset, batch_size = test_batch_size, shuffle=FALSE)

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(38, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )

        self.inter_fc = nn.Sequential(
            nn.Linear(256, 2),
            nn.Softmax()
        )

        self.train_fc = nn.Sequential(
            # nn.Linear(256, 256),
            # nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
            nn.Softmax()
        )

        self.__init_parameters()

        self.pretrain = None

    def __init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

    def __freeze_vars(self, module_name):
        assert module_name in ["features", "inter_fc", "train_fc"]
        for n, v in self.named_modules():
            if module_name in n:
                if isinstance(v, nn.Linear):
                    print("__freeze_vars: ", n)
                    v.weight.requires_grad = False
                    v.bias.requires_grad = False

    def __unfreeze_vars(self, module_name):
        assert module_name in ["features", "inter_fc", "train_fc"]
        for n, v in self.named_modules():
            if module_name in n:
                if isinstance(v, nn.Linear):
                    print("__unfreeze_vars: ", n)
                    v.weight.requires_grad = True
                    v.bias.requires_grad = True

    def set_pretrain(self, v = True):
        self.pretrain = v
        if self.pretrain == True:
            self.__freeze_vars("train_fc")
            self.__unfreeze_vars("features")
            self.__unfreeze_vars("inter_fc")
        else:
            self.__unfreeze_vars("train_fc")
            self.__freeze_vars("features")
            self.__freeze_vars("inter_fc")


    def forward(self, x):
        if self.pretrain == None:
            print("Error: please set stage!")
            exit(-1)

        x = self.features(x)
        if self.pretrain == TRUE:
            x = self.inter_fc(x)
        else:
            x = self.train_fc(x)

        return x

device = torch.device('cuda:0')
net = MLP().to(device)
print(net)
# criteon = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8])).to(device)
criteon = nn.CrossEntropyLoss().to(device)

# scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
#                                            milestones=[10], gamma=0.1)

all_result = {}
all_result['pretrain_acc'] = []
all_result['transfer_acc'] = []
pretrain_best_acc = 0
pretrain_cur_acc = 0
transfer_best_acc = 0
transfer_cur_acc = 0

if "pretrain" in stages:
    ###################### pretrain stage #########################
    net.set_pretrain()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                            milestones=[15000, 20000], gamma=0.1)
    total_iter = 30000
    cur_iter = 0
    test_iter = 300
    print("pretrain....")
    for (data, target) in inter_train_loader:
        data, target = data.to(device), target.to(device)
        # data, target = data, target
        logits = net(data.to(torch.float32))
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        cur_iter+=1
        if cur_iter % 50 == 0:
            loss_print = loss / train_batch_size * 100
            print('Train iter:[{}/{} ({:.0f}%)] class1 distribution:[{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                cur_iter, total_iter,
                    100. * cur_iter / total_iter, target.sum(), len(target),
                    100. * target.sum() / len(target), loss_print))

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
                # print("total one: ", total_one)
                test_loss /= train_batch_size * 100
                test_loss *= 100
                pretrain_cur_acc = (100. * correct / (train_batch_size * 100)).item()
                all_result["pretrain_acc"].append(pretrain_cur_acc)
                if pretrain_cur_acc > pretrain_best_acc:
                    pretrain_best_acc = pretrain_cur_acc
                    save_ckpt(save_path, "pretrain_best", net, optimizer, scheduler, all_result)

                print('\nTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%) Best Acc {:.2f}%\n'.format(
                    test_loss, correct, (train_batch_size * 100), pretrain_cur_acc, pretrain_best_acc))
                data_visualization(save_path, all_result, "pretrain_acc")
                file.flush()

        if cur_iter >= total_iter:
            break

    save_ckpt(save_path, "pretrain_final", net, optimizer, scheduler, all_result)

###################### transfer stage #########################
if "transfer" in stages:
    checkpoint_path = osp.join(pretrain_path, "checkpoints", "pretrain_best.pt")
    checkpoint = torch.load(checkpoint_path, map_location = torch.device('cuda:0'))
    net.load_state_dict(checkpoint['model'])

    net.set_pretrain(False)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                            milestones=[int(epochs * 0.5), int(epochs * 0.75)], gamma=0.1)
    print("trainsfer....")

    for epoch in range(epochs):
        net.train()
        batch_idx=0
        total_one = 0
        for (data, target) in train2_loader:
            data, target = data.to(device), target.to(device)
            # data, target = data, target
            logits = net(data.to(torch.float32))
            loss = criteon(logits, target)

            optimizer.zero_grad()
            loss.backward()
            # print(w1.grad.norm(), w2.grad.norm())
            optimizer.step()
            batch_idx+=1
            # if batch_idx % 5 == 0:
            loss_print = loss / train_batch_size * 100
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(concat_dataset),
                        100. * batch_idx * len(data) / len(concat_dataset), loss_print))
            total_one += target.sum()
        # print("train total one: ", total_one)
        scheduler.step()
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
            # print("total one: ", total_one)
            test_loss /= len(sub_test_dataset)
            test_loss *= 100

            transfer_cur_acc = (100. * correct / len(sub_test_dataset)).item()
            all_result["transfer_acc"].append(transfer_cur_acc)
            if transfer_cur_acc > transfer_best_acc:
                transfer_best_acc = transfer_cur_acc
                save_ckpt(save_path, "transfer_best", net, optimizer, scheduler, all_result)

            print('\nTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%) Best Acc {:.2f}%\n'.format(
                test_loss, correct, len(sub_test_dataset), transfer_cur_acc, transfer_best_acc))
            
            data_visualization(save_path, all_result, "transfer_acc")
        file.flush()

    save_ckpt(save_path, "transfer_final", net, optimizer, scheduler, all_result)
    


file.close()