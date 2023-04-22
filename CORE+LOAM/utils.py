# @Time: 2023/01
# @Author: Heeyoon Yang
# @Email: yooonyblooming@gmail.com


import numpy as np
import torch
import pickle
import random
import networkx as nx


def get_metric_scores(scores, targets, k, ht_dict, ht_idxs, eval):
    # eval : hit, mrr, cov, arp, tail, tailcov
    head_idxs = ht_idxs[0]
    tail_idxs = ht_idxs[1]
    sub_scores = scores.topk(k)[1]
    sub_scores = sub_scores.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()

    cur_hits, cur_mrrs = [], []
    for score, target in zip(sub_scores, targets):
        isin = np.isin(score, target)
        if sum(isin) == 1:
            cur_hits.append(True)
            cur_mrrs.append(1 / (np.where(isin == True)[0][0] + 1))
        else:
            cur_hits.append(False)
            cur_mrrs.append(0)
    
    # overall acc and coverage
    eval[0] += cur_hits
    eval[1] += cur_mrrs
    eval[2] += np.unique(sub_scores).tolist()
    
    # coverage for head and tail
    score_head_idxs = np.where(np.isin(sub_scores, ht_dict['head']))
    score_tail_idxs = np.where(~np.isin(sub_scores, ht_dict['head']))

    # head acc and cov
    eval[3] += np.array(cur_hits)[head_idxs].tolist()
    eval[4] += np.array(cur_mrrs)[head_idxs].tolist()
    eval[5] += np.unique(sub_scores[score_head_idxs[0], score_head_idxs[1]]).tolist()
    
    # tail acc and cov
    eval[6] += np.array(cur_hits)[tail_idxs].tolist()
    eval[7] += np.array(cur_mrrs)[tail_idxs].tolist()
    eval[8] += [np.mean(np.sum(~np.isin(sub_scores, ht_dict['head']), axis=1) / k)]  # Tail@K
    eval[9] += np.unique(sub_scores[score_tail_idxs[0], score_tail_idxs[1]]).tolist() # Tail Coverage@K
    return eval



def data_check(all_usr_pois, mode, max_len=200, pad_item=0):
    if mode == 'train':
        us_lens = [len(upois) for upois in all_usr_pois]
        us_len_max = max(us_lens)
        len_max = min(us_len_max, max_len)
    else:
        len_max = max_len
    
    # us_pois, us_seqs, us_lens = [], [], []
    us_pois, us_lens = [], []
    for upois in all_usr_pois:
        new_upois = upois[-len_max:]
        le = len(new_upois)
        new_seqs = new_upois + [pad_item] * (len_max-le)
        us_pois.append(new_seqs)
        us_lens.append(le)
    
    return np.array(us_pois), np.array(us_lens), len_max


def build_global_graph(dataset_name, ht_dict, type='all_mean', rand=random.Random(220)):
    print("Building Global Graph....", end=" ")
    head_rels = pickle.load(open(f'../../Datasets/{dataset_name}/head_relations.pickle', 'rb'))
    tail_rels = pickle.load(open(f'../../Datasets/{dataset_name}/tail_relations.pickle', 'rb'))
        
    if dataset_name == 'nowplaying':
        sample_num = 12
    elif dataset_name == 'yoochoose_4':
        sample_num = 34
    elif dataset_name == 'yoochoose_64':
        sample_num = 8
    elif dataset_name == 'diginetica':
        sample_num = 11 
    elif dataset_name == 'retailrocket':
        sample_num = 11
    
    
    ### to make popular node have less edges
    new_rels = []
    for i in range(len(head_rels)):
        try:
            new_rels.append(np.array(rand.sample(head_rels[i].tolist(), sample_num)))
        except:
            new_rels.append(head_rels[i])
    relation = np.concatenate(new_rels + tail_rels)   
    global_graph = nx.from_edgelist(relation, create_using=nx.DiGraph)
    weight_dict = pickle.load(open(f'../../Datasets/{dataset_name}/pop_inv_dict.pickle', 'rb'))
    
    for node in global_graph.nodes():
        global_graph.nodes[node]['weight'] = weight_dict[node]
    
    print("Done.")        
    return global_graph


def NWA(batch_graph, targets, max_len, total_n_walks, alpha=0.3, walk_len=10, n_walks=5, rand=random.Random(220)):
    walks = []
    aug_targets = []
    for tar in targets:
        for i in range(n_walks):
            cur_node = tar
            path = [cur_node]
            while len(path) <= walk_len:
                neighbors = np.array([[src, batch_graph.nodes(data='weight')[src]] for src, dst in batch_graph.in_edges(cur_node)])
                if len(neighbors) > 0:
                    if rand.random() >= alpha:
                        cur_node = neighbors[np.argmax(neighbors[:, 1])][0]
                    else:
                        cur_node = rand.choice(neighbors[:, 0])
                    path.append(cur_node)
                else:
                    break
            if len(path) > 2:
                aug_targets.append(path[0])
                walks.append(list(reversed(path[1:])))

    aug_sess_pois = [sess + [0]*(max_len-len(sess)) for sess in walks]
    aug_sess_pois = np.array(aug_sess_pois)
    aug_targets = np.array(aug_targets)
    
    if len(aug_sess_pois) > total_n_walks:
        idxs = list(range(len(walks)))
        sampled_idx = rand.sample(idxs, total_n_walks)
        aug_sess_pois = aug_sess_pois[sampled_idx]
        aug_targets = aug_targets[sampled_idx]
    
    return aug_sess_pois, aug_targets

class Data():
    def __init__(self, data, opt, ht_dict, max_len, mode, shuffle=False):
        inputs = data[0]
        self.inputs, self.inputs_len, self.len_max = data_check(inputs, mode, max_len)
        self.targets = np.array(data[1])
        self.length = len(self.inputs)
        self.ht_dict = ht_dict
        self.shuffle = shuffle
        if mode == 'train':
            self.nwa = True
            self.alpha = opt.random_walk_alpha
            self.global_graph = build_global_graph(opt.dataset, self.ht_dict, opt.edge_drop_type)
            self.aug_sampled_ratio = opt.aug_ratio
            self.n_aug = []
        else:
            self.nwa = False
        

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.targets = self.targets[shuffled_arg]

        n_batch = int(self.length / batch_size)
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices.append(np.arange(n_batch * batch_size, self.length))
        return slices

    def get_slice(self, i):
        inputs, targets = self.inputs[i], self.targets[i]
        head_idxs = np.where(np.isin(targets, self.ht_dict['head']))[0].tolist()
        tail_idxs = np.setdiff1d(np.arange(len(inputs)), head_idxs)
        org_inputs_len = len(inputs)
          
        if self.nwa:
            aug_starts = np.random.choice(targets, int(len(inputs) * self.aug_sampled_ratio))
            inputs_nodes = np.unique(np.concatenate([inputs[np.nonzero(inputs)], targets]))
            batch_graph = nx.subgraph(self.global_graph, inputs_nodes)
            aug_inputs, aug_targets = NWA(batch_graph, aug_starts, self.len_max, int(len(inputs) * self.aug_sampled_ratio), self.alpha)
            self.n_aug.append(len(aug_inputs))
            
            if aug_inputs.shape != (0,):
                inputs = np.concatenate([inputs, aug_inputs], axis=0)
                targets = np.concatenate([targets, aug_targets], axis=0)
            
        ## torch Long 
        inputs = torch.LongTensor(inputs)
        targets = torch.LongTensor(targets)
        return inputs, targets, [head_idxs, tail_idxs], org_inputs_len


class EarlyStopping:
    def __init__(self, save_dir, n_items, n_head_items, n_tail_items, wandb, patience=10):
        self.save_dir = save_dir
        self.patience = patience
        self.n_items = n_items
        self.n_head_items = n_head_items
        self.n_tail_items = n_tail_items
        self.wandb = wandb
        self.counter = 0
        self.best_scores = None
        self.best_epoch = None
        self.early_stop = False
    
    def scoring(self, score):
        for evals in score:
            # hit, mrr, cov, 
            evals[0] = np.mean(evals[0]) * 100
            evals[1] = np.mean(evals[1]) * 100
            evals[2] = len(np.unique(evals[2])) / self.n_items * 100
            # head hit, mrr, cov
            evals[3] = np.mean(evals[3]) * 100
            evals[4] = np.mean(evals[4]) * 100
            evals[5] = len(np.unique(evals[5])) / self.n_head_items * 100
            # tail hit, mrr, tail, tail cov
            evals[6] = np.mean(evals[6]) * 100
            evals[7] = np.mean(evals[7]) * 100
            evals[8] = np.mean(evals[8]) * 100
            evals[9] = len(np.unique(evals[9])) / self.n_tail_items * 100
        return score
    
    def score_print(self, score):
        eval10 = score[0]
        eval20 = score[1]
        
        self.wandb.log({"HR@10" : eval10[0], "MRR@10" : eval10[1], "Cov@10" : eval10[2],
                        "HRh@10" : eval10[3], "MRRh@10" : eval10[4], "HCov@10" : eval10[5],
                        "HRt@10": eval10[6], "MRRt@10" : eval10[7], "Tail@10" : eval10[8], "TCov@10" : eval10[9],
                        "HR@20" : eval20[0], "MRR@20" : eval20[1], "Cov@20" : eval20[2],
                        "HRh@20" : eval20[3], "MRRh@20" : eval20[4], "HCov@20" : eval20[5],
                        "HRt@20": eval20[6], "MRRt@20" : eval20[7], "Tail@20" : eval20[8], "TCov@20" : eval20[9],
                        })

        print('Metric\t\tHR@10\tMRR@10\tCov@10\tHRh@10\tMRRh@10\tHCov@10\tHRt@10\tMRRt@10\tTail@10\tTCov@10')
        print(f'Value\t\t'+'\t'.join(format(eval, ".2f") for eval in eval10))
        print('Metric\t\tHR@20\tMRR@20\tCov@20\tHRh@20\tMRRh@20\tHCov@20\tHRt@20\tMRRt@20\tTail@20\tTCov@20')
        print(f'Value\t\t' + '\t'.join(format(eval, ".2f") for eval in eval20))
    
    
    def compare(self, score, epoch):
        # compare score based on only HR@20
        if score[1][0] > self.best_scores[1][0]:
            ## update best scores
            self.best_scores = score
            self.best_epoch = epoch
            return False
        else:
            ## not update = count as early_stop
            return True

    def best_score_print(self):
        print("-"* 100)
        print('Best Result\tHR@10\tMRR@10\tCov@10\tHRh@10\tMRRh@10\tHCov@10\tHRt@10\tMRRt@10\tTail@10\tTCov@10\tEpoch')
        print(f'Value\t\t' + '\t'.join(format(result, ".2f") for result in self.best_scores[0]) + f'\t{self.best_epoch}')
        print('Best Result\tHR@20\tMRR@20\tCov@20\tHRh@20\tMRRh@20\tHCov@20\tHRt@20\tMRRt@20\tTail@20\tTCov@20\tEpoch')
        print(f'Value\t\t' + '\t'.join(format(result, ".2f") for result in self.best_scores[1]) + f'\t{self.best_epoch}')
    
    def save_checkpoint(self, model, epoch):
        model_save_name =  f"{self.save_dir}/ep{epoch}.pt"
        torch.save(model.state_dict(), model_save_name)
        print(f"Best Model Saved at {model_save_name}")
        
    def __call__(self, model, score, epoch):
        new_score = self.scoring(score)
        if self.best_scores is None:
            self.best_scores = new_score
            self.best_epoch = epoch
            self.save_checkpoint(model, epoch)
            
        else:
            flag = self.compare(score, epoch)
            if flag:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.save_checkpoint(model, epoch)
                self.counter = 0
        
        self.score_print(new_score)
        self.best_score_print()
        
def init_seed(seed=220):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False