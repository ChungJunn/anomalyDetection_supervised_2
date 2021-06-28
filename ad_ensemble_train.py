import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import pickle as pkl

import math
import sys
import time

import argparse

from ad_model import RNN_enc_RNN_clf
from ad_ensemble_data import AD_RNN_Dataset

from sklearn.metrics import classification_report

import torch
import torch.nn as nn
from torch.nn import functional as F

from EnsemblePytorch.torchensemble import VotingClassifier
from EnsemblePytorch.torchensemble.utils.logging import set_logger

def sequencify(input, label, ids, rnn_len):
    xlist = []
    ylist = []

    for idx in ids:
        data_range = range((idx-rnn_len+1),(idx+1))
        x_data = input[data_range,:,:]
        y_data = label[idx]
        xlist.append(x_data)
        ylist.append(y_data)
    
    x_datas = np.stack(xlist).astype(np.float32)
    y_datas = np.array(ylist).astype(np.int64)

    return x_datas, y_datas

def train_main(args):
    if args.classifier == 'rnn':
        test_dnn = False
    else:
        test_dnn = True

    train = AD_RNN_Dataset(mode="train",
                           csv_path=args.csv_path,
                           ids_path=args.ids_path,
                           stat_path=args.stat_path,
                           data_name=args.data_name,
                           rnn_len=args.rnn_len,
                           test_dnn=test_dnn)
    valid = AD_RNN_Dataset(mode="valid",
                           csv_path=args.csv_path,
                           ids_path=args.ids_path,
                           stat_path=args.stat_path,
                           data_name=args.data_name,
                           rnn_len=args.rnn_len,
                           test_dnn=test_dnn)
    test = AD_RNN_Dataset(mode="test",
                           csv_path=args.csv_path,
                           ids_path=args.ids_path,
                           stat_path=args.stat_path,
                           data_name=args.data_name,
                           rnn_len=args.rnn_len,
                           test_dnn=test_dnn)
    
    device = torch.device('cuda')

    criterion = torch.nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=args.batch_size, shuffle=True)
    
    test_x, test_y = sequencify(input=test.input, 
                                label=test.label,
                                ids=test.ids,
                                rnn_len=args.rnn_len)
    
    # Set the Logger
    logger = set_logger(args.exp_name)
    estimator_args = {'args':args}

    if args.encoder == "rnn" and args.classifier == "rnn":
        model = VotingClassifier(
            estimator=RNN_enc_RNN_clf,
            n_estimators=args.n_estimators,
            estimator_args=estimator_args,
            cuda=True,
        )
        if args.dataset == 'cnsm_exp2_1':
            epochs=20
        else:
            epochs=10

    # Set the optimizer
    model.set_optimizer('Adam', lr=1e-3)

    # Train and Evaluate
    model.fit(
        train_loader,
        epochs=epochs,
        test_loader=valid_loader,
        criterion=criterion,
    )

    preds = model.predict(test_x)
    preds = torch.argmax(preds, dim=1)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(test_y, preds)
    prec = precision_score(test_y, preds)
    rec = recall_score(test_y, preds)
    f1 = f1_score(test_y, preds)

    print('{} | acc: {:.4f} | prec: {:.4f} | rec: {:.4f} | f1: {:.4f} |'.format("test", acc, prec, rec, f1))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # task
    parser.add_argument('--label', type=str)
    # exp_name
    parser.add_argument('--exp_name', type=str)
    # dataset
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dim_input', type=int)
    parser.add_argument('--rnn_len', type=int)
    parser.add_argument('--csv_path', type=str)
    parser.add_argument('--ids_path', type=str)
    parser.add_argument('--stat_path', type=str)
    parser.add_argument('--dict_path', type=str)
    parser.add_argument('--data_name', type=str)
    # feature mapping
    parser.add_argument('--dim_feature_mapping', type=int)

    # enc
    parser.add_argument('--encoder', type=str)
    parser.add_argument('--nlayer', type=int)
    # dnn-enc
    parser.add_argument('--dim_enc', type=int)
    # rnn-enc
    parser.add_argument('--bidirectional', type=int)
    parser.add_argument('--dim_lstm_hidden', type=int)
    # transformer-enc
    parser.add_argument('--nhead', type=int)
    parser.add_argument('--dim_feedforward', type=int)
    # readout
    parser.add_argument('--reduce', type=str)
    parser.add_argument('--dim_att', type=int)

    # clf
    parser.add_argument('--classifier', type=str)
    parser.add_argument('--clf_n_lstm_layers', type=int)
    parser.add_argument('--clf_n_fc_layers', type=int)
    parser.add_argument('--clf_dim_lstm_hidden', type=int)
    parser.add_argument('--clf_dim_fc_hidden', type=int)
    parser.add_argument('--clf_dim_output', type=int)

    # training parameter
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--patience', type=float)
    parser.add_argument('--drop_p', type=float)
    
    # ensemble parameter
    parser.add_argument('--n_estimators', type=int)

    args = parser.parse_args()
    params = vars(args)

    train_main(args)