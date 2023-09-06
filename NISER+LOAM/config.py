import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    # for basic
    parser.add_argument('--dataset', default='yoochoose_64', help='dataset name: tmall/diginetica/30music/retailrocket/nowplaying/')
    parser.add_argument('--max_len', type=int, default=200)
    parser.add_argument('--seed', type=int, default=220)
    
    # for wandb
    parser.add_argument('--entity', default=None, type=str, help='wandb user name')
    parser.add_argument('--project', default=None, type=str, help='wandb project name')
    parser.add_argument('--name', default=None, type=str, help='wandb model name')
    
    # for model
    parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
    parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
    parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
    parser.add_argument('--validation', action='store_true', help='validation')
    parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
    parser.add_argument('--norm', default=True, help='adapt NISER, l2 norm over item and session embedding')
    parser.add_argument('--TA', default=False, help='use target-aware or not')
    parser.add_argument('--scale', default=True, help='scaling factor sigma')
    
    # for training
    parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--device', type=str, default='gpu', help='gpu/cpu')
    parser.add_argument('--gpu_num', type = int, default=3, help = 'cuda number')
    parser.add_argument('--model_save', type=bool, default=True)
    parser.add_argument('--model_save_path', type=str, default=None)
    
    # for NWA 
    parser.add_argument('--aug_ratio', type=float, default=0.1)
    parser.add_argument('--edge_drop_type', type=str, default='all_mean')
    parser.add_argument('--random_walk_alpha', default=0.5, type=float)
    
    # for TSM
    parser.add_argument('--mixup_lam', type=float, default=0.8)
    parser.add_argument('--sim_type', type=str, default='cosine', help='cosine/jaccard/random')
    parser.add_argument('--misc', type=bool, default=False)
    return parser.parse_args()

opt = parse_args()