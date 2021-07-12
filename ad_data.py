import pandas as pd
import numpy as np
import pickle as pkl
import torch
from torch.utils.data import Dataset
import sys
from ad_utils import get_const

class AD_SUP2_RNN_ITERATOR2:
    def __init__(self, mode, csv_path, ids_path, stat_path, dict_path, data_name, batch_size, rnn_len, test_dnn, label):
        # load csv, ids
        df_data = pd.read_csv(csv_path)
        np_data = np.array(df_data)
        self.data = np_data[:,:-3].astype(np.float32)

        if label == 'sla':
            label_i = -3
            self.label = np_data[:,label_i]
        # label encoding
        elif label == 'rcl':
            label_i = -1
            self.label = np_data[:,label_i]

            # import dict
            with open(dict_path, 'rb') as fp:
                d = pkl.load(fp)
            class2idx = d['class2idx']

            # encode
            for n in range(len(self.label)):
                self.label[n] = class2idx[self.label[n]]
            self.label = self.label.astype(np.int64)
        else:
            print('label must be either sla or rcl')
            sys.exit(0)

        with open(ids_path, 'rb') as fp:
            ids = pkl.load(fp)
        if mode == 'train':
            self.ids = ids['train']
        if mode == 'valid':
            self.ids = ids['valid']
        if mode == 'test':
            self.ids = ids['test']

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

        # split to nodes
        n_nodes, n_features = get_const(data_name)
        self.n_nodes = n_nodes
        self.n_features = n_features

        # prepare lists
        label_col = "SLA_Label"
        datas = []
        headers = []

        for i in range(self.n_nodes):
            start, end = (i * self.n_features), ((i+1) * self.n_features)
            vnf_data = self.data[:, start:end]
            datas.append(np.copy(vnf_data))

        ## replace with add tvt to the dataset paths
        self.input = np.stack(datas)
        self.input = self.input.transpose(1,0,2) # (Bn x V x D)

        self.ids_i = 0
        self.n_samples = self.input.shape[0]
        self.rnn_len = rnn_len
        self.batch_size = batch_size
        self.n_ids = len(self.ids)
        self.test_dnn = test_dnn

    def reset(self):
        self.ids_i = 0
        return

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_ids

    def __next__(self):
        x_data = np.zeros((self.batch_size, self.rnn_len, self.n_nodes, self.n_features))
        y_data = np.zeros((self.batch_size))
        end_of_data = 0

        b_len = 0
        for i in range(self.batch_size):
            # end-condition
            if (self.ids_i + i) >= self.n_ids:
                self.reset()
                end_of_data = 1
                # break

            # get idx
            idx = self.ids[self.ids_i + i]

            # get segment and label
            data_range = range((idx-self.rnn_len+1),(idx+1))
            x_data[i,:,:,:] = self.input[data_range,:,:]
            y_data[i] = self.label[idx] 
            b_len += 1

        x_data = x_data[:b_len,:,:,:] # (Bn x Tx x V x D)
        y_data = y_data[:b_len]

        self.ids_i += b_len

        x_data, y_data = torch.tensor(x_data).type(torch.float32), torch.tensor(y_data).type(torch.int32)

        if self.test_dnn == True:
            x_data = x_data[:,-1,:,:] # (Bn x V x D)

        return x_data, y_data, end_of_data

class AD_RNN_Dataset(Dataset):
    def __init__(self, mode, csv_path, ids_path, stat_path, data_name, rnn_len, test_dnn):
         # load csv, ids
        df_data = pd.read_csv(csv_path)
        np_data = np.array(df_data)
        self.data = np_data[:,:-3].astype(np.float32)

        # assume sla label
        label_i = -3
        self.label = np_data[:,label_i].astype(np.int64)

        if mode == 'plot':
            self.ids = list(range((rnn_len-1), len(self.data)))
        else:
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

        n_nodes, n_features = get_const(data_name)
        self.n_nodes = n_nodes
        self.n_features = n_features
        
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
            x_data = x_data[-1:,:,:] # (Bn x V x D)

        return x_data, y_data

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tvt', type=str, default='sup_train')
    parser.add_argument('--csv_path', type=str, default='/home/chl/autoregressor/data/raw/cnsm_exp2_1_data.csv')
    parser.add_argument('--ids_path', type=str, default='/home/chl/autoregressor/data/cnsm_exp2_1_data/indices.rnn_len16.pkl')
    parser.add_argument('--stat_path', type=str, default='/home/chl/autoregressor/data/raw/cnsm_exp2_1_data.csv.stat')
    parser.add_argument('--data_name', type=str, default='cnsm_exp2_1_data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--rnn_len', type=int, default=16)
    args = parser.parse_args()

    iter = AD_SUP2_RNN_ITERATOR2(tvt=args.tvt, csv_path=args.csv_path, ids_path=args.ids_path, stat_path=args.stat_path, data_name=args.data_name, batch_size=args.batch_size, rnn_len=args.rnn_len)

    for iloop, (anno, label, end_of_data) in enumerate(iter):
        anno, label = anno.to(device), label.to(device)
        # print('from iterator: ', anno.shape, label.shape)
        # take hidden, obtain output and loss, fix the model
