# @Time: 2023/01
# @Author: Heeyoon Yang
# @Email: yooonyblooming@gmail.com

import os
import time
import datetime
import pickle

import wandb
import torch

from config import opt
from model import *
from utils import *

print(opt)
init_seed(opt.seed)
wandb.init(project=f"{opt.project}", entity={opt.entity}, name={opt.name})
wandb.config.update(opt)

device_name = 'cpu' if opt.device == 'cpu' else f"cuda:{opt.gpu_num}"
device = torch.device(device_name)

if opt.model_save == True:
    model_save_path = f'save_models/{opt.dataset}/{datetime.datetime.now().strftime("%m%d_%H%M%S")}'
    os.makedirs(model_save_path, exist_ok=True)

def main():
    train_data = pickle.load(open(f'../../Datasets/{opt.dataset}/train.txt', 'rb'))
    test_data = pickle.load(open(f'../../Datasets/{opt.dataset}/test.txt', 'rb'))
    n_items = pickle.load(open(f'../../Datasets/{opt.dataset}/n_node.txt', 'rb'))
    ht_dict = pickle.load(open(f'../../Datasets/{opt.dataset}/ht_dict.pickle', 'rb'))
    
    train_data = Data(train_data, opt, ht_dict, opt.max_len, mode='train', shuffle=True)
    test_data = Data(test_data, opt, ht_dict, train_data.len_max, mode='test', shuffle=False)
    
    model = CORE(opt, device, n_items).to(device)
    earlystop = EarlyStopping(model_save_path, n_items, len(ht_dict['head']), len(ht_dict['tail']), wandb)

    start = time.time()
    for epoch in range(opt.epoch):
        print('-' * 100)
        print('Epoch: ', epoch)
        loss, results = train_test(model, train_data, test_data, n_items, ht_dict, wandb)
        print(f"average # of augmented samples : {np.mean(train_data.n_aug):.2f}")
        earlystop(model, results, epoch)
        if earlystop.early_stop:
            break

    print('-' * 100)
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()