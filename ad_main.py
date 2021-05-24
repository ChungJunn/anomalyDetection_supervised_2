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

from ad_model import AD_SUP2_MODEL3, AD_SUP2_MODEL6
from ad_data import AD_SUP2_RNN_ITERATOR2
from ad_test import eval_forward, eval_binary, get_valid_loss, log_neptune

from sklearn.metrics import classification_report

def call_model(args, device):
    if args.encoder == 'transformer' and args.classifier == 'dnn':
        model = AD_SUP2_MODEL3(args)
    elif args.encoder == 'rnn' and args.classifier == 'rnn':
        model = AD_SUP2_MODEL6(args)
    
    model = model.to(device)

    return model

def train_main(args, neptune):
    device = torch.device('cuda:0')
    criterion = F.nll_loss

    model = call_model(args, device)

    if args.classifier == 'rnn':
        test_dnn = False
    else:
        test_dnn = True

    trainiter = AD_SUP2_RNN_ITERATOR2(mode='train',
                                      csv_path=args.csv_path,
                                      ids_path=args.ids_path,
                                      stat_path=args.stat_path,
                                      dict_path=args.dict_path,
                                      data_name=args.data_name,
                                      batch_size=args.batch_size,
                                      rnn_len=args.rnn_len,
                                      test_dnn=test_dnn,
                                      label=args.label)
    validiter = AD_SUP2_RNN_ITERATOR2(mode='valid',
                                      csv_path=args.csv_path,
                                      ids_path=args.ids_path,
                                      stat_path=args.stat_path,
                                      dict_path=args.dict_path,
                                      data_name=args.data_name,
                                      batch_size=args.batch_size,
                                      rnn_len=args.rnn_len,
                                      test_dnn=test_dnn,
                                      label=args.label)
    testiter = AD_SUP2_RNN_ITERATOR2(mode='test',
                                     csv_path=args.csv_path,
                                     ids_path=args.ids_path,
                                     stat_path=args.stat_path,
                                     dict_path=args.dict_path,
                                     data_name=args.data_name,
                                     batch_size=args.batch_size,
                                     rnn_len=args.rnn_len,
                                     test_dnn=test_dnn,
                                     label=args.label)

    print('trainiter: {} samples'.format(len(trainiter)))
    print('validiter: {} samples'.format(len(validiter)))
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
    best_valid_loss = None
    savedir = './result/' + args.out_file
    if args.label == 'rcl':
        with open(args.dict_path, 'rb') as fp:
            idx2class = (pkl.load(fp))['idx2class']

    for ei in range(args.max_epoch):
        for li, (x_data, y_data, end_of_data) in enumerate(trainiter):
            x_data = x_data.to(dtype=torch.float32, device=device)
            y_data = y_data.to(dtype=torch.int64, device=device)

            optimizer.zero_grad()

            output = model(x_data)

            # go through loss function
            loss = criterion(output, y_data)
            loss.backward()

            # optimizer
            optimizer.step()
            train_loss1 += loss.item()
            train_loss2 += loss.item()

            if end_of_data == 1: break

            if (args.batch_size == 1) and (log_idx % log_interval) == (log_interval - 1):
                print('{:d} | {:d} | {:.4f}'.format(ei+1, log_idx+1, train_loss1/log_interval))
                if neptune is not None: neptune.log_metric('train loss (n_samples)', log_idx+1, train_loss1/log_interval)
                train_loss1 = 0.0
            log_idx += 1

        train_loss2 = train_loss2 / (li + 1)
        print('epoch: {:d} | train_loss: {:.4f}'.format(ei+1, train_loss2))
        if neptune is not None: neptune.log_metric('train_loss2', ei+1, train_loss2)
        train_loss2 = 0.0

        valid_loss = get_valid_loss(model, validiter, device)
        print('epoch: {:d} | valid_loss: {:.4f}'.format(ei+1, valid_loss))
        if neptune is not None: neptune.log_metric('valid_loss', ei+1, valid_loss)

        if ei == 0 or valid_loss < best_valid_loss:
            torch.save(model, savedir)
            bc = 0
            best_valid_loss = valid_loss
            print('found new best model')
        else:
            bc += 1
            if bc > args.patience:
                print('early stopping..')
                break
            print('bad counter == %d' % (bc))

    model = torch.load(savedir)
    
    # evaluation
    eval_modes = ['valid', 'test']
    for eval_mode in eval_modes:
        dataiter = eval(eval_mode + 'iter')
        targets, preds = eval_forward(model, dataiter, device)

        # for binary class
        if args.label == 'sla':
            # metric
            acc, prec, rec, f1 = eval_binary(targets, preds)

            # std and neptune
            print('{} | acc: {:.4f} | prec: {:.4f} | rec: {:.4f} | f1: {:.4f} |'.format(eval_mode, acc, prec, rec, f1))
            if neptune is not None:
                neptune.set_property(eval_mode + ' acc', acc)
                neptune.set_property(eval_mode + ' prec', prec)
                neptune.set_property(eval_mode + ' rec', rec)
                neptune.set_property(eval_mode + ' f1', f1)

        # for multi-class
        elif args.label == 'rcl':
            result_dict = classification_report(targets, preds, target_names=idx2class, output_dict=True)

            # std and neptune
            print(eval_mode, result_dict)
            if neptune is not None:
                log_neptune(result_dict, eval_mode, neptune)

    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # task
    parser.add_argument('--label', type=str)
    parser.add_argument('--use_neptune', type=int)
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

    if args.use_neptune == 1:
        neptune.init('cjlee/apnoms2021')
        experiment = neptune.create_experiment(name=args.exp_name, params=params)
        args.out_file = experiment.id + '.pth'
    else:
        neptune=None
        args.out_file = 'dummy.pth'

    print('parameters:')
    print('='*90)
    print(params)
    print('='*90)

    train_main(args, neptune)
