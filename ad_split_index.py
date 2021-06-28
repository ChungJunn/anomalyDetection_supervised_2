import numpy as np
import argparse
import math
import pickle as pkl
import pandas as pd
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tpi_train')
    parser.add_argument('--rnn_len', type=int, default=16)
    parser.add_argument('--ratios', type=str, default='[0.65, 0.1, 0.25]')
    args = parser.parse_args()

    base_dir = '/home/chl/autoregressor/data/raw/'

    # load data for number of data
    data = pd.read_csv(base_dir + args.dataset + '_data.csv')

    # obtain indices and ratios 
    ids = list(range(len(data)))[args.rnn_len:]
    n_samples = len(ids)

    if args.dataset == "cnsm_exp1" or args.dataset == "cnsm_exp2_1" or args.dataset == "cnsm_exp2_2":
        # split indices and make dicts
        [tr_ratio, val_ratio, test_ratio] = eval(args.ratios)
        tr_idx = math.ceil(tr_ratio * n_samples)
        val_idx = math.ceil((tr_ratio+val_ratio) * n_samples)

        # shuffle
        np.random.shuffle(ids)
        
        tr_ids = ids[:tr_idx]
        val_ids = ids[tr_idx:val_idx]
        test_ids = ids[val_idx:]

        out_dict = {'train':tr_ids, 'valid':val_ids, 'test':test_ids}
    
    elif args.dataset == "tpi_train":
        # split indices and make dicts
        [tr_ratio, val_ratio] = [0.9, 0.1]
        tr_idx = math.ceil(tr_ratio * n_samples)

        # shuffle
        np.random.shuffle(ids)
        
        tr_ids = ids[:tr_idx]
        val_ids = ids[tr_idx:]

        out_dict = {'train':tr_ids, 'valid':val_ids}
    
    else:
        print("--dataset must be either cnsm or tpi_train")

    target_base_dir = '/home/chl/autoregressor/data/'

    path = target_base_dir + args.dataset + '_data/'
    if not os.path.exists(path):
        os.mkdir(target_base_dir + args.dataset + '_data/')
   
    target_path = target_base_dir + args.dataset + '_data/' + 'indices.rnn_len'+ str(args.rnn_len) + '.pkl'
    
    with open(target_path, 'wb') as fp:
        pkl.dump(out_dict, fp)