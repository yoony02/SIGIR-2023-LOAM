import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    # for basic 
    parser.add_argument('--dataset', default='yoochoose_64', type = str)
    parser.add_argument('--max_len', default=50, type=int)
    parser.add_argument('--seed', default=220)
    
    # for wandb
    parser.add_argument('--entity', default=None, type=str, help='wandb user name')
    parser.add_argument('--project', default=None, type=str, help='wandb project name')
    parser.add_argument('--name', default=None, type=str, help='wandb model name')
    
    # for model
    parser.add_argument('--type', default='trm', help='ave / trm')
    parser.add_argument('--hidden_size', default=50, type=int)
    parser.add_argument('--inner_size', default=256, type=int)
    parser.add_argument('--n_layers', default=2, type=int)
    parser.add_argument('--n_heads', default=1, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
    parser.add_argument('--lr_dc', type=float, default=0.1)
    parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
    parser.add_argument('--layer_norm_eps', type=float, default=1e-12)
    parser.add_argument('--temperature', type=float, default=0.07)
    
    # for training
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--device', default='gpu', type=str, help='gpu/cpu')
    parser.add_argument('--gpu_num', default=0, type=int)
    parser.add_argument('--model_save', type=bool,  default=True, help='saving model')
    parser.add_argument('--model_save_path', default=None)
    
    # for NWA
    parser.add_argument('--aug_ratio', type=float, default=0.1)
    parser.add_argument('--edge_drop_type', type=str, default='all_mean')
    parser.add_argument('--random_walk_alpha', default=0.5, type=float)
    
    # for TSM
    parser.add_argument('--mixup_lam', type=float, default=0.8)
    parser.add_argument('--sim_type', type=str, default='cosine', help='cosine/jaccard/random')
    return parser.parse_args()

opt = parse_args()