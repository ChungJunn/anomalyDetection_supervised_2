import pandas as pd
import numpy as np
import torch

# create annotation and adjacency matrices and dataloader
class ad_gnn_iterator:
    def __init__(self, tvt, data_dir, csv_files, direction, recur_w):
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
        self.n_samples = self.node_features[0].shape[0]
        self.n_node_features = self.node_features[0].shape[1]

        # initialize the variables
        self.n_nodes = len(self.node_features)
        self.direction = direction
        self.recur_w = recur_w

    def make_annotation_matrix(self, idx):
        # initialize the matrix
        annotation = np.zeros([self.n_nodes, self.n_node_features])

        # retrieve the related data using idx
        for ni in range(self.n_nodes):
            for fi in range(self.n_node_features):
                annotation[ni,fi] = (self.node_features[ni])[idx, fi]

        return annotation

    def make_adj_matrix(self, idx, direction='forward', recur_w=0.7):
        # initialize the matrix
        A_in = np.zeros([self.n_nodes, self.n_nodes])
        A_out = np.zeros([self.n_nodes, self.n_nodes])

        n_edges = 1.0 if direction=='forward' else 2.0
        edge_weight = (1 - recur_w) / n_edges

        import math # retrieve the related data using idx
        for from_node in range(self.n_nodes): # no edge feature for last node
            A_in[from_node, from_node] = recur_w

            if from_node < (self.n_nodes - 1):
                A_in[from_node, from_node + 1] = edge_weight

            if direction == 'bi-direction' and from_node > 0:
                A_in[from_node, from_node - 1] = edge_weight

        A_out += A_in

        return A_in, A_out

    def reset(self):
        self.idx = 0
        return

    def __next__(self):
        end_of_data=0

        if self.idx >= (self.n_samples - 1):
            end_of_data=1
            self.reset()

        annotation = self.make_annotation_matrix(self.idx)
        A_in, A_out = self.make_adj_matrix(self.idx, direction=self.direction,recur_w=self.recur_w)

        label = self.label[self.idx]

        self.idx += 1

        annotation = torch.tensor(annotation)

        A_in = torch.tensor(A_in)
        A_out = torch.tensor(A_out)
        label = torch.tensor(label)

        return annotation, A_in, A_out, label, end_of_data

    def __iter__(self):
        return self

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--direction', type=str, help='', default='forward')
    parser.add_argument('--n_nodes', type=int)
    parser.add_argument('--csv1', type=str)
    parser.add_argument('--csv2', type=str)
    parser.add_argument('--csv3', type=str)
    parser.add_argument('--csv4', type=str)
    parser.add_argument('--csv5', type=str)
    parser.add_argument('--csv_label', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--recur_w', type=float)
    args = parser.parse_args()

    csv_files=[]
    for n in range(1, args.n_nodes+1):
        csv_file = eval('args.csv' + str(n))
        csv_files.append(csv_file)
    csv_files.append(args.csv_label)

    iter = ad_gnn_iterator(tvt='sup_train', data_dir=args.data_dir, csv_files=csv_files, direction=args.direction, recur_w=args.recur_w)

    for iloop, (anno, A_out, A_in, label, end_of_data) in enumerate(iter):
        print(iloop, anno.shape, A_out.shape, A_in.shape, label.shape)
        print(label)
        import pdb; pdb.set_trace()
