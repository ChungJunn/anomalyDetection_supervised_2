'''
adopted from pytorch.org (Classifying names with a character-level RNN-Sean Robertson)
'''
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
import neptune

from ad_model import AD_SUP2_MODEL6
from ad_model import AD_SUP2_MODEL7
from ad_data import AD_SUP2_RNN_ITERATOR2
from ad_test import eval_main, test

def train_main(args, neptune):
    device = torch.device('cuda:0')
    criterion = F.nll_loss
        
    model = AD_SUP2_MODEL6(args).to(device)

    if args.classifier == 'rnn':
        test_dnn = False
    else:
        test_dnn = True

    trainiter = AD_SUP2_RNN_ITERATOR2(mode='train',
                                      batch_size=args.batch_size,
                                      rnn_len=args.rnn_len,
                                      test_dnn=False,
                                      label='False')
    validiter = AD_SUP2_RNN_ITERATOR2(mode='valid',
                                      batch_size=args.batch_size,
                                      rnn_len=args.rnn_len,
                                      test_dnn=False,
                                      label='False')
    testiter = AD_SUP2_RNN_ITERATOR2(mode='test',
                                      batch_size=args.batch_size,
                                      rnn_len=args.rnn_len,
                                      test_dnn=False,
                                      label='False')

    print('trainiter: {} samples'.format(len(trainiter)))
    print('validiter: {} samples'.format(len(valiter)))
    print('testiter: {} samples'.format(len(testiter)))

    # declare optimizer
    estring = "optim." + args.optimizer
    optimizer = eval(estring)(model.parameters(), lr=args.lr)

    # modify the dataset to produce labels
    # create a training loop
    train_loss1 = 0.0
    train_loss2 = 0.0
    log_interval=1000
    log_idx=0
    bc = 0
    best_val_f1 = None
    savedir = './result/' + args.out_file
    n_samples = trainiter.n_samples
    clf_hidden = None

    for ei in range(args.max_epoch):
        for li, (anno, label, end_of_data) in enumerate(trainiter):
            anno = anno.to(dtype=torch.float32, device=device)
            label = label.to(dtype=torch.int64, device=device)

            optimizer.zero_grad()

            output = model(anno)

            # go through loss function
            loss = criterion(output, label)
            loss.backward()

            # optimizer
            optimizer.step()
            train_loss1 += loss.item()
            train_loss2 += loss.item()

            if end_of_data == 1: break

            if (log_idx % log_interval) == (log_interval - 1):
                print('{:d} | {:d} | {:.4f}'.format(ei+1, log_idx+1, train_loss1/log_interval))
                if neptune is not None: neptune.log_metric('train loss (n_samples)', log_idx+1, train_loss1/log_interval)
                train_loss1 = 0.0
            log_idx += 1

        train_loss2 = train_loss2 / li
        print('epoch: {:d} | train_loss: {:.4f}'.format(ei+1, train_loss2))
        if neptune is not None: neptune.log_metric('train loss2', ei, train_loss2)
        train_loss2 = 0.0

        acc,prec,rec,f1=eval_main(model,valiter,device)
        print('epoch: {:d} | val_f1: {:.4f}'.format(ei+1, f1))
        if neptune is not None: neptune.log_metric('valid f1', ei, f1)

        if ei == 0 or f1 > best_val_f1:
            torch.save(model, savedir)
            bc = 0
            best_val_f1=f1
            print('found new best model')
        else:
            bc += 1
            if bc > args.patience:
                print('early stopping..')
                break
            print('bad counter == %d' % (bc))

    model = torch.load(savedir)
    
    datasets = ['cnsm_exp1', 'cnsm_exp2_1', 'cnsm_exp2_2']
    for dset in datasets:
        acc, prec, rec, f1 = test(model, dset, args.batch_size, args.rnn_len, test_dnn, device)

        print('{} | acc: {:.4f} | prec: {:.4f} | rec: {:.4f} | f1: {:.4f}'
              .format(dset, acc, prec, rec, f1))

        if neptune is not None:
            neptune.set_property(dset+'_acc', acc)
            neptune.set_property(dset+'_prec', prec)
            neptune.set_property(dset+'_rec', rec)
            neptune.set_property(dset+'_f1', f1)

    return

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
    parser.add_argument('--use_feature_mapping', type=int)
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
    parser.add_argument('--max_epoch', type=int)

    args = parser.parse_args()
    params = vars(args)

    #neptune.init('cjlee/apnoms2021')
    #experiment = neptune.create_experiment(name=args.exp_name, params=params)
    #args.out_file = experiment.id + '.pth'

    neptune=None
    args.out_file = 'dummy.pth'

    print('parameters:')
    print('='*90)
    print(params)
    print('='*90)

    train_main(args, neptune)
