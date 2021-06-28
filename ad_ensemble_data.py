
# write dataloader
import torch
from torch.utils.data import Dataset

import os
import pandas as pd
import numpy as np
import pickle as pkl

class AD_RNN_Dataset(Dataset):
    def __init__(self, mode, csv_path, ids_path, stat_path, data_name, rnn_len, test_dnn):
         # load csv, ids
        df_data = pd.read_csv(csv_path)
        np_data = np.array(df_data)
        self.data = np_data[:,:-3].astype(np.float32)

        # assume sla label
        label_i = -3
        self.label = np_data[:,label_i].astype(np.int64)

        with open(ids_path, 'rb') as fp:
            ids = pkl.load(fp)
        if mode == 'train':
            self.ids = ids['train']
        elif mode == 'valid':
            self.ids = ids['valid']
        elif mode == 'test':
            self.ids = ids['test']
        else:
            print('mode must be either train, valid, or test')

        # normalize
        if mode == 'train':
            self.x_avg = np.mean(self.data, axis=0)
            self.x_std = np.std(self.data, axis=0)

            for i in range(len(self.x_std)):
                if self.x_std[i] == 0:
                    self.x_std[i] = 0.001

            fp = open(stat_path, 'w')
            for i in range(self.x_avg.shape[0]):
                if i > 0:
                    fp.write(', ')
                fp.write('%.9f' % (self.x_avg[i]))
            fp.write('\n')
            for i in range(self.x_std.shape[0]):
                if i > 0:
                    fp.write(', ')
                fp.write('%.9f' % (self.x_std[i]))
            fp.write('\n')
            fp.close()
        else:
            fp = open(stat_path, 'r')
            lines = fp.readlines()
            self.x_avg= np.asarray([float(s) for s in lines[0].split(',')])
            self.x_std= np.asarray([float(s) for s in lines[1].split(',')])
            fp.close()
        
        self.data -= np.expand_dims(self.x_avg, axis=0)
        self.data /= np.expand_dims(self.x_std, axis=0)

        self.n_features = 23
        # split to nodes
        if data_name == 'cnsm_exp1_data':
            vnfs = ['fw', 'ids', 'flowmon', 'dpi', 'lb']
            self.n_nodes = 5
        elif data_name == 'cnsm_exp2_1_data' or data_name == 'cnsm_exp2_2_data':
            vnfs = ['fw', 'flowmon', 'dpi', 'ids']
            self.n_nodes = 4
        else:
            print('data_name must be cnsm_exp1_data, cnsm_exp2_1_data, or cnsm_exp2_2_data')
            import sys; sys.exit(-1)

        # prepare lists
        label_col = "SLA_Label"
        datas = []
        headers = []

        for i in range(self.n_nodes):
            start, end = (i * self.n_features), ((i+1) * self.n_features)
            vnf_data = self.data[:, start:end]
            datas.append(np.copy(vnf_data))

        ## replace with add tvt to the dataset paths
        self.input = np.stack(datas).astype(np.float32)
        self.input = self.input.transpose(1,0,2) # (Bn x V x D)

        self.ids_i = 0
        self.n_samples = self.input.shape[0]
        self.rnn_len = rnn_len
        self.n_ids = len(self.ids)
        self.test_dnn = test_dnn

    def __len__(self):
        return self.n_ids

    def __getitem__(self, idx):
        idx = self.ids[idx]

        # get segment and label
        data_range = range((idx-self.rnn_len+1),(idx+1))
        x_data = self.input[data_range,:,:]
        y_data = self.label[idx]

        if self.test_dnn == True:
            x_data = x_data[-1,:,:] # (Bn x V x D)

        return x_data, y_data

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--csv_path', type=str, default='/home/chl/autoregressor/data/raw/cnsm_exp1_data.csv')
    parser.add_argument('--ids_path', type=str, default='/home/chl/autoregressor/data/cnsm_exp1_data/indices.rnn_len16.pkl')
    parser.add_argument('--stat_path', type=str, default='/home/chl/autoregressor/data/raw/cnsm_exp1_data.csv.stat')
    parser.add_argument('--data_name', type=str, default='cnsm_exp1_data')
    parser.add_argument('--rnn_len', type=int, default=16)
    parser.add_argument('--test_dnn', type=int, default=1)
    args = parser.parse_args()

    device = torch.device('cuda')

    train = AD_RNN_Dataset(mode=args.mode,
                           csv_path=args.csv_path,
                           ids_path=args.ids_path,
                           stat_path=args.stat_path,
                           data_name=args.data_name,
                           rnn_len=args.rnn_len,
                           test_dnn=args.test_dnn)

    train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)

    n_samples = 0

    for iloop, (anno, label) in enumerate(train_loader):
        anno, label = anno.to(device), label.to(device)
        # print('from iterator: ', anno.shape, label.shape)
        # take hidden, obtain output and loss, fix the model
        n_samples += label.shape[0]
        
    print('end of data')
    import pdb; pdb.set_trace()