import numpy as np
import torch
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

from ad_data import AD_SUP2_RNN_ITERATOR2

def eval_main(model, validiter, device):
    model.eval()

    preds = []
    targets = []
    clf_hidden = None

    # forward the whole dataset and obtain result
    for li, (anno, ys, end_of_data) in enumerate(validiter):
        anno = anno.to(dtype=torch.float32, device=device)
        ys = ys.to(dtype=torch.int64, device=device)

        #outs, clf_hidden = model(anno, clf_hidden)
        outs = model(anno)

        outs = outs.detach().cpu().numpy()
        ys = ys.detach().cpu().numpy().reshape(-1,1)

        preds.append(outs)
        targets.append(ys)

        if end_of_data == 1: break

    import pdb; pdb.set_trace()

    # obtain results using metrics
    preds = np.vstack(preds)
    targets = np.vstack(targets)

    preds = np.argmax(preds, axis=1)

    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds)
    rec = recall_score(targets, preds)
    f1 = f1_score(targets, preds)

    model.train()
    return acc, prec, rec, f1

def test(model, dataset, batch_size, rnn_len, test_dnn, device):
    # call dataiter
    base_dir = os.environ['HOME'] + '/autoregressor/data/'
    csv_path = base_dir + 'raw/' + dataset + '_data.csv' 
    ids_path = base_dir + dataset + '_data/indices.rnn_len16.pkl'
    stat_path = csv_path + '.stat'
    data_name = dataset + '_data'
    rnn_len = 16

    testiter = AD_SUP2_RNN_ITERATOR2(tvt='sup_test', csv_path=csv_path, ids_path=ids_path, stat_path=stat_path, data_name=data_name, batch_size=batch_size, rnn_len=rnn_len, test_dnn=test_dnn)

    # evaluate
    acc, prec, rec, f1 = eval_main(model, testiter, device)

    return acc, prec, rec, f1

if __name__ == '__main__':
    device = torch.device('cuda')
    from ad_model import AD_SUP2_MODEL3
    model = AD_SUP2_MODEL3(22, 2, 128, 'self-attention', 1, 22, 1, 22).to(device)

    dataset = 'cnsm_exp2_2'

    batch_size = 32



