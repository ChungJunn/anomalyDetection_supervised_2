import pandas as pd
import numpy as np
import pickle as pkl
import torch

class AD_SUP2_RNN_ITERATOR2:
    def __init__(self, tvt, csv_path, ids_path, stat_path, data_name, batch_size, rnn_len, test_dnn=True):
        # load csv, ids
        df_data = pd.read_csv(csv_path)
        np_data = np.array(df_data)
        
        self.data = np_data[:,:-3].astype(np.float32)
        self.label = np_data[:,-3].astype(np.int64)
        
        with open(ids_path, 'rb') as fp:
            ids = pkl.load(fp)
        if tvt == 'sup_train':
            self.ids = ids['train']
        if tvt == 'sup_val':
            self.ids = ids['valid']
        if tvt == 'sup_test':
            self.ids = ids['test']

        # normalize
        if tvt == 'sup_train':
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
        if data_name == 'cnsm_exp1_data':
            vnfs = ['fw', 'ids', 'flowmon', 'dpi', 'lb']
            self.n_nodes = 5
            self.n_features = 23
        elif data_name == 'cnsm_exp2_1_data' or data_name == 'cnsm_exp2_2_data':
            vnfs = ['fw', 'flowmon', 'dpi', 'ids']
            self.n_nodes = 4
            self.n_features = 23
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
        self.input = np.stack(datas)
        self.input = self.input.transpose(1,0,2)

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

    def __next__(self):
        x_data = np.zeros((self.batch_size, self.rnn_len, self.n_nodes, self.n_features)) # T B E
        y_data = np.zeros((self.batch_size)) # T B E
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

        x_data = x_data[:b_len,:,:,:]
        y_data = y_data[:b_len]

        self.ids_i += b_len

        x_data, y_data = torch.tensor(x_data).type(torch.float32), torch.tensor(y_data).type(torch.int32)

        if self.test_dnn == True:
            x_data = x_data[:,-1,:,:]
        else:
            x_data = x_data.squeeze()

        return x_data, y_data, end_of_data

class AD_SUP2_RNN_ITERATOR:
    def __init__(self, tvt, data_dir, pkl_files, batch_size):
        ## replace with add tvt to the dataset paths
        pkl_paths=[]

        for n in range(len(pkl_files)):
            pkl_path=data_dir+tvt+'.'+pkl_files[n]
            pkl_paths.append(pkl_path)

        # iter for n_nodes
        self.node_features=[]
        for n in range(len(pkl_paths)-1):
            with open(pkl_paths[n], 'rb') as fp:
                node_data = pkl.load(fp)
                self.node_features.append(node_data)
        with open(pkl_paths[-1], 'rb') as fp:
            self.label= pkl.load(fp)

        self.input = np.stack(self.node_features)
        self.input = self.input.transpose(1,2,0,3)

        self.idx = 0
        self.n_samples = self.input.shape[0]
        self.n_node_features = self.input.shape[-1]
        self.n_nodes = len(pkl_paths) - 1
        self.rnn_len = self.input.shape[1]

    def reset(self):
        self.idx=0
        return

    def __iter__(self):
        return self

    def __next__(self):
        x_data = np.zeros((self.rnn_len, self.n_nodes, self.n_node_features))
        y_data = np.zeros((1,))
        end_of_data = 0

        x_data[:,:,:] = self.input[self.idx]
        y_data[:] = self.label[self.idx]
        self.idx+=1

        if self.idx >= (self.n_samples-1):
            end_of_data = 1
            self.reset()

        x_data = torch.tensor(x_data).type(torch.float32)
        y_data = torch.tensor(y_data).type(torch.int64)

        return x_data, y_data, end_of_data

class AD_SUP2_DNN_ITERATOR:
    def __init__(self, tvt, data_dir, pkl_files, batch_size):
        ## replace with add tvt to the dataset paths
        pkl_paths=[]

        for n in range(len(pkl_files)):
            pkl_path=data_dir+tvt+'.'+pkl_files[n]
            pkl_paths.append(pkl_path)

        # iter for n_nodes
        self.node_features=[]
        self.n_nodes = len(pkl_paths) - 1

        for n in range(len(pkl_paths)-1):
            with open(pkl_paths[n], 'rb') as fp:
                node_data = pkl.load(fp)
                self.node_features.append(node_data)

        with open(pkl_paths[-1], 'rb') as fp:
            self.label= pkl.load(fp)

        self.input = np.stack(self.node_features)
        self.input = self.input.transpose(1,2,0,3)
        self.input = self.input[:,-1,:,:]
        # node x bsz x seq_len x data_dim
        
        '''
        if self.n_nodes == 4:
            self.input = np.concatenate((self.input[:,0,:],self.input[:,1,:],self.input[:,2,:],self.input[:,3,:]), axis=1)
        elif self.n_nodes == 5:
            self.input = np.concatenate((self.input[:,0,:],self.input[:,1,:],self.input[:,2,:],self.input[:,3,:], self.input[:,4,:]), axis=1)
        else:
            print('node number must be either 4 or 5')
        '''

        # prepare some properties
        self.idx = 0
        self.n_samples = self.input.shape[0]
        self.n_node_features = self.input.shape[-1]
        self.batch_size = batch_size

    def reset(self):
        self.idx=0
        return

    def __iter__(self):
        return self

    def __next__(self):
        x_data = np.zeros((self.batch_size, self.n_nodes, self.n_node_features)) # T B E
        y_data = np.zeros((self.batch_size,))
        end_of_data = 0

        if self.idx >= self.n_samples:
            self.reset()
            end_of_data=1

        b_len = 0
        for i in range(self.batch_size):
            if self.idx+i >= self.n_samples:
                break

            x_data[i,:,:] = self.input[self.idx+i,:,:]
            y_data[i] = self.label[self.idx+i]
            b_len += 1

        x_data = x_data[:b_len]
        y_data = y_data[:b_len]
        self.idx += self.batch_size

        x_data, y_data = torch.tensor(x_data).type(torch.float32), torch.tensor(y_data).type(torch.int32)
        return x_data, y_data, end_of_data # B E

# create annotation and adjacency matrices and dataloader
class AD_SUP2_ITERATOR:
    def __init__(self, tvt, data_dir, csv_files, batch_size):
        ## replace with add tvt to the dataset paths
        csv_paths=[]

        for n in range(len(csv_files)):
            csv_path=data_dir+tvt+'.'+csv_files[n]
            csv_paths.append(csv_path)

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        # iteration for n_nodes
        self.node_features=[]
        for n in range(len(csv_paths)-1):
            pp_data = scaler.fit_transform(np.array(pd.read_csv(csv_paths[n])))
            self.node_features.append(pp_data)
        self.label = np.array(pd.read_csv(csv_paths[-1]))

        self.idx = 0
        self.batch_size=batch_size
        self.n_samples = self.node_features[0].shape[0]
        self.n_node_features = self.node_features[0].shape[1]

        # initialize the variables
        self.n_nodes = len(self.node_features)
    
    def get_status(self):
        print(self.idx, self.batch_size, self.n_samples, self.n_node_features)

    def make_annotation_matrix(self, idx):
        # initialize the matrix
        annotation = np.zeros([self.n_nodes, self.n_node_features])

        # retrieve the related data using idx
        for ni in range(self.n_nodes):
            for fi in range(self.n_node_features):
                annotation[ni,fi] = (self.node_features[ni])[idx, fi]

        return annotation

    def reset(self):
        self.idx = 0
        return

    def __next__(self):
        x_data = np.zeros((self.batch_size, self.n_nodes, self.n_node_features))
        y_data = np.zeros((self.batch_size,))
        end_of_data=0

        # b_size
        del_i=0
        for bi in range(self.batch_size):
            if self.idx+bi >= (self.n_samples - 1):
                end_of_data=1
                self.reset()
                del_i=0

            x_data[bi, :, :] = self.make_annotation_matrix(self.idx+bi)
            y_data[bi] = self.label[self.idx+bi]
            del_i+=1

        self.idx += del_i

        x_data = torch.tensor(x_data).type(torch.float32)
        y_data = torch.tensor(y_data).type(torch.int64)
        
        return x_data, y_data, end_of_data

    def __iter__(self):
        return self

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
