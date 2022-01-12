import pickle
from torch.utils.data import DataLoader,Dataset
import torch
import numpy as np

class MyDataset(Dataset):

    def __init__(self,data):
        self.data=data
        if(data=="train"):
            with open('./dataset/pkl/train_features.pkl', 'rb') as file:
                train = pickle.load(file)
            with open('./dataset/pkl/train_labels.pkl', 'rb') as file:
                label = pickle.load(file)
            self.train_data = torch.tensor(np.array(train))
            self.train_label = torch.tensor(np.array(label))
            self.len = len(train)
        elif(data=="train_inter"):
            with open('./dataset/pkl/inter_train_features.pkl', 'rb') as file:
                train = pickle.load(file)
            with open('./dataset/pkl/inter_train_labels.pkl', 'rb') as file:
                label = pickle.load(file)
            self.train_data = torch.tensor(np.array(train))
            self.train_label = torch.tensor(np.array(label))
            self.len = len(train)

        elif(data=="test"):
            with open('./dataset/pkl/test_features.pkl', 'rb') as file:
                test = pickle.load(file)
                self.test_data = torch.tensor(np.array(test))
                self.len = len(test)
        self.class0_set = []
        self.class1_set = []
        self.__get_class_set()
    def __get_class_set(self):
        for i, c in enumerate(self.train_label.data):
            if c == 0:
                self.class0_set.append(i)
            else:
                self.class1_set.append(i)

    def __getitem__(self, index):
        if self.data=="train" or self.data=="train_inter":
            return self.train_data[index],self.train_label[index]
        else:
            return self.test_data[index]

    def __len__(self):
        if self.data == "train" or self.data == "train_inter":
            return len(self.train_data)
        else:
            return len(self.test_data)
