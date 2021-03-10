import pandas as pd
import numpy as np
import torch

class AD_SUP2_RNN_ITERATOR:
    def __init__(self, tvt, data_dir, pkl_files, batch_size):
        ## replace with add tvt to the dataset paths
        pkl_paths=[]

        for n in range(len(pkl_files)):
            pkl_path=data_dir+tvt+'.'+pkl_files
            pkl_paths.append(pkl_path)

        # iteration for n_nodes
        self.node_features=[]
        for n in range(len(pkl_paths)-1):
            with open(pkl_path[n], 'rb') as fp:
                node_data = pkl.load(fp)
                self.node_features.append(node_data)
        self.label = np.array(pd.read_csv(csv_paths[-1]))

    def make_annotation_matrix(self, idx):

    def reset(self):

    def __next__(self):

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

    parser.add_argument('--n_nodes', type=int)
    parser.add_argument('--csv1', type=str)
    parser.add_argument('--csv2', type=str)
    parser.add_argument('--csv3', type=str)
    parser.add_argument('--csv4', type=str)
    parser.add_argument('--csv5', type=str)
    parser.add_argument('--csv_label', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--reduce', type=str)
    parser.add_argument('--batch_size', type=int)
    args = parser.parse_args()

    csv_files=[]
    for n in range(1, args.n_nodes + 1):
        csv_file = eval('args.csv' + str(n))
        csv_files.append(csv_file)
    csv_files.append(args.csv_label)

    iter = AD_SUP2_ITERATOR(tvt='sup_train', data_dir=args.data_dir, csv_files=csv_files, batch_size=args.batch_size)
    device = torch.device('cuda')

    from ad_model import AD_SUP2_MODEL4
    import torch.nn.functional as F

    model = AD_SUP2_MODEL4(
            dim_input=22,
            enc_dim_lstm_hidden=64,
            reduce='max',
            bidirectional=1,
            use_feature_mapping=1,
            dim_feature_mapping=24,
            nlayer=2,
            clf_dim_lstm_hidden=64,
            clf_dim_fc_hidden=128,
            clf_dim_output=2).to(device)

    hidden = None
    for iloop, (anno, label, end_of_data) in enumerate(iter):
        anno, label = anno.to(device), label.to(device)
        # print('from iterator: ', anno.shape, label.shape)
        # take hidden, obtain output and loss, fix the model
        out, hidden = model(anno, hidden)
        out = out.squeeze(0)

        loss = F.nll_loss(out, label)

        import pdb; pdb.set_trace()
        if end_of_data==1: break

    print('iter status: ', iter.get_status())
