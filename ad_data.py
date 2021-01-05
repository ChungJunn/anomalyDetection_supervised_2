import pandas as pd
import numpy as np
import torch

# create annotation and adjacency matrices and dataloader
class AD_SUP2_ITERATOR:
    def __init__(self, tvt, data_dir, csv_files):
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
        end_of_data=0

        if self.idx >= (self.n_samples - 1):
            end_of_data=1
            self.reset()

        annotation = self.make_annotation_matrix(self.idx)
        label = self.label[self.idx]

        self.idx += 1

        annotation = torch.tensor(annotation).type(torch.float32)
        label = torch.tensor(label).type(torch.int64)

        return annotation, label, end_of_data

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
    args = parser.parse_args()

    csv_files=[]
    for n in range(1, args.n_nodes+1):
        csv_file = eval('args.csv' + str(n))
        csv_files.append(csv_file)
    csv_files.append(args.csv_label)

    iter = AD_SUB2_ITERATOR(tvt='sup_train', data_dir=args.data_dir, csv_files=csv_files)

    from ad_model import AD_SUP2_MODEL1

    model = AD_SUP2_MODEL1(reduce=args.reduce)
    print(model)

    for iloop, (anno, label, end_of_data) in enumerate(iter):
        print(iloop, anno.shape, label.shape)
        print(label)

        logits = model(anno)
