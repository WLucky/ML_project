import math
import torch
import torch.utils.data as tordata


class AverageSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_shuffle = batch_shuffle

    def __iter__(self):
        while True:
            sample_indices = []
            # pid --> label
            class0_list = random_sample_list(
                self.dataset.class0_set, self.batch_size//2)

            class1_list = random_sample_list(
                self.dataset.class1_set, self.batch_size//2)

            sample_indices = class0_list + class1_list
            # 第三次只是打乱顺序
            if self.batch_shuffle:
                sample_indices = random_sample_list(
                    sample_indices, len(sample_indices))

            yield sample_indices

    def __len__(self):
        return len(self.dataset)


def random_sample_list(obj_list, k):
    # Returns a random permutation of integers from 0 to n - 1.
    idx = torch.randperm(len(obj_list))[:k]
    if torch.cuda.is_available():
        idx = idx.cuda()
    idx = idx.tolist()
    return [obj_list[i] for i in idx]

