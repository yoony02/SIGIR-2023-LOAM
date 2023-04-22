# @Time: 2023/01
# @Author: Heeyoon Yang
# @Email: yooonyblooming@gmail.com

"""
Reference: 
    Yupeng Hou et al. "CORE: Simple and Effective Session-based Recommendation within Consistent Representation Space."
    In SIGIR 2022.
    
Reference code:
    The authors' Pytorch implementation: https://github.com/RUCAIBox/CORE
"""


import time
import datetime
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from utils import get_metric_scores

def jac(a, b):
    unique = set(np.concatenate([a, b]))
    inter = np.intersect1d(a, b)
    return len(inter) / len(unique)

class MultiHeadAttention(Module):
    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FeedForward(Module):
    
    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": nn.functional.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states





class TransformerLayer(Module):
    def __init__(self, n_heads, hidden_size, inner_size, dropout, hidden_act='gelu', eps=1e-12):
        super(TransformerLayer, self).__init__()
        self.multi_head_attn = MultiHeadAttention(n_heads, hidden_size, dropout, dropout, eps)
        self.feed_forward = FeedForward(hidden_size, inner_size, dropout, hidden_act, eps)

    def forward(self, hidden_state, attn_mask):
        attn_out = self.multi_head_attn(hidden_state, attn_mask)
        feed_forward_out = self.feed_forward(attn_out)
        return feed_forward_out



class TransformerEncoder(Module):
    def __init__(self, n_layers=2, n_heads=2, hidden_size=64, inner_size=256, dropout=0.5):
        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(n_heads, hidden_size, inner_size, dropout)
        self.layer_module = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
    
    def forward(self, hidden_state, attn_mask, output_all_encode_layers=True):
        all_encode_layers = []
        for layer in self.layer_module:
            hidden_state = layer(hidden_state, attn_mask)
            if output_all_encode_layers:
                all_encode_layers.append(hidden_state)
        if not output_all_encode_layers:
            all_encode_layers.append(hidden_state)
        return all_encode_layers




class CORE(Module):
    def __init__(self, opt, device, n_items):
        super(CORE, self).__init__()
        self.batch_size = opt.batch_size
        self.n_items = n_items
        self.max_len = opt.max_len
        self.hidden_size = opt.hidden_size
        self.dropout = opt.dropout
        self.temperature = opt.temperature
        self.type = opt.type
        self.device = device
        self.mixup_lam = opt.mixup_lam
        self.sim_type = opt.sim_type

        if self.type == 'trm':
            self.inner_size = opt.inner_size
            self.n_layers = opt.n_layers
            self.n_heads = opt.n_heads
            self.eps = opt.layer_norm_eps

            self.pos_embedding = nn.Embedding(self.max_len, self.hidden_size)
            self.trm_encoder = TransformerEncoder(self.n_layers, self.n_heads, self.hidden_size, self.inner_size, self.dropout)
            self.layer_norm = nn.LayerNorm(self.hidden_size, eps=self.eps)
            self.linear = nn.Linear(self.hidden_size, 1)


        self.embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.sess_dropout = nn.Dropout(opt.dropout)
        self.item_dropout = nn.Dropout(opt.dropout)
    
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self._reset_parameters()

    
    def _reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    
    def session_interpolate(self, items, targets, seq_hidden, tail_idxs, org_inputs_len):
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
            aug_seq_hiddens = seq_hidden[tail_idxs].unsqueeze(-1)
            full_seq_hiddens = seq_hidden[:org_inputs_len].unsqueeze(0).expand(len(tail_idxs), -1, -1)
            sims =  torch.matmul(full_seq_hiddens, aug_seq_hiddens)
            mixup_sess_srcs = torch.argsort(sims.squeeze(-1), descending=True)[:, 1]
        
        # mixup two session
        # target are same as original one
        mixed_seq_hidden = self.mixup_lam * seq_hidden[tail_idxs] + (1 - self.mixup_lam) * seq_hidden[mixup_sess_srcs]
        ys = targets[tail_idxs]
                
        return mixed_seq_hidden, ys
    

    def ave_net(self, seq):
        mask = seq.gt(0)
        alpha = mask.to(torch.float) / mask.sum(dim=-1, keepdim=True)
        return alpha.unsqueeze(-1)
        
    
    def trm_net(self, seq):
        mask = seq.gt(0)

        pos_idx = torch.arange(seq.size(1), dtype=torch.long).to(self.device)
        pos_idx = pos_idx.unsqueeze(0).expand_as(seq)
        pos_emb = self.pos_embedding(pos_idx)

        sess_emb = self.embedding(seq)
        input_emb = sess_emb + pos_emb
        input_emb = self.layer_norm(input_emb)
        input_emb = self.sess_dropout(input_emb)

        # get attention mask
        attn_mask = (seq != 0)
        ext_attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        ext_attn_mask = torch.where(ext_attn_mask, 0., -10000.)

        trm_output = self.trm_encoder(input_emb, ext_attn_mask, output_all_encode_layers=True)
        output = trm_output[-1]
        
        alpha = self.linear(output).to(torch.double)
        alpha = torch.where(mask.unsqueeze(-1), alpha, -9e15)
        alpha = torch.softmax(alpha, dim=1, dtype=torch.float)
        return alpha


    def encode(self, seq):
        sess_emb = self.embedding(seq)
        sess_emb = self.sess_dropout(sess_emb)

        if self.type == 'ave':
            alpha = self.ave_net(seq)
        elif self.type == 'trm':
            alpha = self.trm_net(seq)
        output = torch.sum(alpha * sess_emb, dim=1)
        output = F.normalize(output, dim=-1)
        return output

    def compute_scores(self, seqs_hidden):
        item_emb = self.embedding.weight
        item_emb = self.item_dropout(item_emb)
        item_emb = F.normalize(item_emb, dim=-1)
        logits = torch.matmul(seqs_hidden, item_emb.transpose(0, 1)) / self.temperature
        return logits


    def forward(self, data, i, mixup=False):
        seqs, targets, ht_idxs, org_inputs_len = data.get_slice(i)
        seqs = seqs.to(self.device)
        targets = targets.to(self.device)
        
        seqs_hidden = self.encode(seqs) 
        
        if mixup:
            mixed_seq_hidden, ys = self.session_interpolate(seqs, targets, seqs_hidden, ht_idxs[1], org_inputs_len)
            mixup_scores = self.compute_scores(mixed_seq_hidden)
            scores = self.compute_scores(seqs_hidden)
            return scores, targets, mixup_scores, ys
        
        else:
            scores = self.compute_scores(seqs_hidden)
            return scores, targets, ht_idxs     



def train_test(model, train_data, test_data, n_items, ht_dict, wandb, Ks=[10, 20]):
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
        