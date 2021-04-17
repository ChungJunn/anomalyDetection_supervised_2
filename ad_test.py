import argparse
import neptune
import torch
import os

from ad_eval import eval_main
from ad_data import AD_SUP2_RNN_ITERATOR2
from ad_model import AD_SUP2_MODEL1

def test(model, dataset, rnn_len, batch_size, device, neptune):
    # obtain data_dir
    base_dir = os.environ['HOME'] + '/autoregressor/data/'
    csv_path = base_dir + 'raw/' + dataset + '_data.csv' 
    ids_path = base_dir + dataset + '_data/indices.rnn_len16.pkl'
    stat_path = csv_path + '.stat'
    data_name = dataset + '_data'
    rnn_len = 16

    testiter = AD_SUP2_RNN_ITERATOR2(tvt='sup_test', csv_path=csv_path, ids_path=ids_path, stat_path=stat_path, data_name=data_name, batch_size=batch_size, rnn_len=rnn_len)

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
