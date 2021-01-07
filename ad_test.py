import argparse
import neptune
import torch

from ad_eval import eval_main
from ad_model import AD_SUP2_MODEL1

'''
eval loads model trained from different datset and measure detection performance in another dataset
'''
if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--n_nodes', type=int)
    parser.add_argument('--csv1', type=str)
    parser.add_argument('--csv2', type=str)
    parser.add_argument('--csv3', type=str)
    parser.add_argument('--csv4', type=str)
    parser.add_argument('--csv5', type=str)
    parser.add_argument('--csv_label', type=str)
    parser.add_argument('--dataset' type=str)

    args = parser.parse_args()
    params = vars(args)

    # set neptune
    neptune.init('cjlee/AnomalyDetection-GNN')
    experiment = neptune.create_experiment(name=args.exp_name, params=params)

    print('parameters:')
    print('='*90)
    print(params)
    print('='*90)

    # load model trained from a given dataset
    device = torch.device('cuda')
    model = torch.load(args.model_path).to(device)

    # load different dataset
    csv_files=[]
    for n in range(1, args.n_nodes+1):
        csv_file=eval('args.csv'+str(n))
        csv_files.append(csv_file)
    csv_files.append(args.csv_label)

    from ad_data import ad_gnn_iterator
    testiter = ad_gnn_iterator(tvt='sup_test', data_dir=args.data_dir, csv_files=csv_files)

    # evaluate the model and measure performance 
    acc, prec, rec, f1 = eval_main(model, testiter, device, neptune=None)

    # print results and logging
    print('acc: {:.4f} | prec: {:.4f} | rec: {:.4f} | f1: {:.4f}'.format(acc, prec, rec, f1))
    neptune.set_property('acc', acc)
    neptune.set_property('prec', prec)
    neptune.set_property('rec', rec)
    neptune.set_property('f1', f1)

