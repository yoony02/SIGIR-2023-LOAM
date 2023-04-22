# @Time: 2023/01
# @Author: Heeyoon Yang
# @Email: yooonyblooming@gmail.com

"""
Reference: 
    Shu Wu et al. "Session-based Recommendation with Graph Neural Networks."
    In AAAI 2019.
    
Reference code:
    The authors' Pytorch implementation: https://github.com/CRIPAC-DIG/SR-GNN
"""


import datetime
import time
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from utils import get_metric_scores
import warnings
warnings.filterwarnings(action = 'ignore')


def jac(a, b):
    unique = set(np.concatenate([a, b]))
    inter = np.intersect1d(a, b)
    return len(inter) / len(unique)


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SRGNN(Module):
    def __init__(self, opt, n_node, device):
        super(SRGNN, self).__init__()
        self.hidden_size = opt.hidden_size
        self.n_node = n_node
        self.device = device
        self.batch_size = opt.batch_size
        self.nonhybrid = opt.nonhybrid
        self.mixup_lam = opt.mixup_lam
        self.sim_type = opt.sim_type
        
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def session_interpolate(self, items, targets, seq_hidden, tail_idxs, org_inputs_len):
        # import pdb; pdb.set_trace()
        if self.sim_type == 'random':
            mixup_sess_srcs = torch.randint(high=seq_hidden.shape[0], size=(len(tail_idxs), )).to(self.device)
           
        # calculate jaccard sim using items 
        elif self.sim_type == "jaccard":
            seqs = items.detach().cpu().numpy()
            tail_seqs = seqs[tail_idxs]
            mixup_sess_srcs = []
            for seq in tail_seqs:
                sims = [jac(seq, seqs[i]) * (-1) for i in range(len(seqs))]
                mixup_sess_srcs.append(np.argsort(sims)[1])
            
        # calculate sim using seq_hidden
        elif self.sim_type == 'cosine':
            # import pdb; pdb.set_trace()
            aug_seq_hiddens = seq_hidden[tail_idxs].unsqueeze(-1)
            full_seq_hiddens = seq_hidden[:org_inputs_len].unsqueeze(0).expand(len(tail_idxs), -1, -1)
            sims =  torch.matmul(full_seq_hiddens, aug_seq_hiddens)
            mixup_sess_srcs = torch.argsort(sims.squeeze(-1), descending=True)[:, 1]
        
        # mixup two session
        # target are same as original one
        mixed_seq_hidden = self.mixup_lam * seq_hidden[tail_idxs] + (1 - self.mixup_lam) * seq_hidden[mixup_sess_srcs]
        ys = targets[tail_idxs]    
        return mixed_seq_hidden, ys
    
    def encode(self, inputs, A, alias_inputs):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        
        mask = alias_inputs.gt(0)
        get = lambda i: hidden[i][alias_inputs[i]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        
        ht = seq_hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(seq_hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * seq_hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))

        return a

    def compute_scores(self, seq_hidden):
        item_hidden = self.embedding.weight[1:]
        scores = torch.matmul(seq_hidden, item_hidden.transpose(1, 0))
        return scores
        
    def forward(self, data, i, mixup=False):
        items, A, alias_inputs, targets, ht_idxs, org_inputs_len = data.get_slice(i)
        items = items.to(self.device)
        A = A.to(self.device)
        alias_inputs = alias_inputs.to(self.device)
        targets = targets.to(self.device)
        
        seq_hidden = self.encode(items, A, alias_inputs)
        
        if mixup:
            mixed_seq_hidden, ys = self.session_interpolate(items, targets, seq_hidden, ht_idxs[1], org_inputs_len)
            mixup_scores = self.compute_scores(mixed_seq_hidden)
            scores = self.compute_scores(seq_hidden)
            return scores, targets, mixup_scores, ys
        else:
            scores = self.compute_scores(seq_hidden)
            return scores, targets, ht_idxs     


def train_test(model, train_data, test_data, n_node, ht_dict, wandb, Ks=[10, 20]):
    epoch_start_train = time.time()
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        scores, targets, mixup_scores, targets_a = model(train_data, i, mixup=True)
        if targets_a.shape == torch.Size([]):
            targets_a = np.array([targets_a])
        h_loss = model.loss_function(scores, targets)
        t_loss = model.loss_function(mixup_scores, targets_a)
        loss = h_loss + t_loss
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()
        if j % 1000 == 0:
            t = time.time() - epoch_start_train
            print('[%d/%d]\tLoss: %.3f  Time: %.2f' % (j, len(slices), loss.item(), t))
            epoch_start_train = time.time()
            
    print('\t\tTotal Loss:\t%.3f' % total_loss)
    wandb.log({"Epoch Train Loss" : total_loss})

    print('start predicting: ', datetime.datetime.now())
    epoch_start_eval = time.time()
    model.eval()
    eval10, eval20 = [[] for i in range(10)], [[] for i in range(10)]
    slices = test_data.generate_batch(model.batch_size)
    with torch.no_grad():
        for i in slices:
            scores, targets, ht_idxs = model(test_data, i, mixup=False)
            eval10 = get_metric_scores(scores, targets, Ks[0], ht_dict, ht_idxs, eval10)
            eval20 = get_metric_scores(scores, targets, Ks[1], ht_dict, ht_idxs, eval20)
    
    t = time.time() - epoch_start_eval
    return loss, [eval10, eval20]



