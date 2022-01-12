import torch
import os.path as osp
import os
import matplotlib.pyplot as plt


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_ckpt(save_path, flag, model, optimizer, scheduler, all_result):
    mkdir(osp.join(save_path, "checkpoints/"))
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'all_result': all_result}
    torch.save(checkpoint,
                osp.join(save_path, 'checkpoints/{}.pt'.format(flag)))

def data_visualization(save_path, all_result, type):
    mkdir(osp.join(save_path, "imgs/"))
    plt.plot(all_result[type], label=type)
    plt.legend()
    plt.savefig(osp.join(save_path, 'imgs/{}_acc.png'.format(type)))
    plt.close()