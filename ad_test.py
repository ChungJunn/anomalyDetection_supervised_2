import argparse
import neptune
import torch

from ad_eval import eval_main
from ad_model import AD_SUP2_MODEL1

'''
eval loads model trained from different datset and measure detection performance in another dataset
'''
'''
if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--trained_dataset', type=str)
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--reduce', type=str)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--dim_lstm_hidden', type=int)
    parser.add_argument('--nlayer', type=int)
    parser.add_argument('--d_model', type=int)
    parser.add_argument('--nhead', type=int)
    parser.add_argument('--dim_feedforward', type=int)
    parser.add_argument('--test_dataset', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--n_nodes', type=int)
    parser.add_argument('--csv1', type=str)
    parser.add_argument('--csv2', type=str)
    parser.add_argument('--csv3', type=str)
    parser.add_argument('--csv4', type=str)
    parser.add_argument('--csv5', type=str)
    parser.add_argument('--csv_label', type=str)

    args = parser.parse_args()
    params = vars(args)

    # set neptune
    neptune.init('cjlee/anomalyDetection-supervised-2')
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

    from ad_data import AD_SUP2_ITERATOR
    testiter = AD_SUP2_ITERATOR(tvt='sup_test', data_dir=args.data_dir, csv_files=csv_files, batch_size=args.batch_size)

    # evaluate the model and measure performance
    acc, prec, rec, f1 = eval_main(model, testiter, device, neptune=None)

    # print results and logging
    print('acc: {:.4f} | prec: {:.4f} | rec: {:.4f} | f1: {:.4f}'.format(acc, prec, rec, f1))
    neptune.set_property('acc', acc)
    neptune.set_property('prec', prec)
    neptune.set_property('rec', rec)
    neptune.set_property('f1', f1)
'''

def test(model, dataset, batch_size, device, neptune):
    # obtain data_dir
    import os
    data_dir = os.environ['HOME'] + '/autoregressor/data/' + dataset + '_data/21.03.10-gnn_data/'

    # load different dataset and create dataloader
    pkls = ['rnn_len16.fw.pkl','rnn_len16.ids.pkl',
            'rnn_len16.flowmon.pkl','rnn_len16.dpi.pkl',
            'rnn_len16.lb.pkl']
    pkl_label = 'rnn_len16.label.pkl'

    if dataset == 'cnsm_exp1':
        pkl_files = pkls
        pkl_files.append(pkl_label)

    elif dataset == 'cnsm_exp2_1' or dataset == 'cnsm_exp2_2':
        pkl_files=[]
        ns = [0,2,3,1] # hard-coding
        for n in ns:
            pkl_files.append(pkls[n])
        pkl_files.append(pkl_label)
    else:
        print('in test(): dataset must be either \'cnsm_exp1\', \'cnsm_exp2_1\', \'cnsm_exp2_2\'')
        import sys; sys.exit(-1)

    from ad_data import AD_SUP2_RNN_ITERATOR
    testiter = AD_SUP2_RNN_ITERATOR(tvt='sup_test', data_dir=data_dir, pkl_files=pkl_files, batch_size=batch_size)

    # evaluate the model and measure performance 
    acc, prec, rec, f1 = eval_main(model, testiter, device, neptune=None)

    # print results and logging
    print('acc: {:.4f} | prec: {:.4f} | rec: {:.4f} | f1: {:.4f}'.format(acc, prec, rec, f1))
    if neptune is not None:
        neptune.set_property('acc', acc)
        neptune.set_property('prec', prec)
        neptune.set_property('rec', rec)
        neptune.set_property('f1', f1)

    return acc, prec, rec, f1

if __name__ == '__main__':
    device = torch.device('cuda')
    from ad_model import AD_SUP2_MODEL3
    model = AD_SUP2_MODEL3(22, 2, 128, 'self-attention', 1, 22, 1, 22).to(device)

    dataset = 'cnsm_exp2_2'

    batch_size = 32
    neptune=None

    test(model, dataset, batch_size, device, neptune)
