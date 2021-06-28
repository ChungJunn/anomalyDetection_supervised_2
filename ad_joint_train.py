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

from ad_model import RNN_enc_RNN_clf
from ad_data import AD_SUP2_RNN_ITERATOR2
from ad_test import eval_forward, eval_binary, get_combined_valid_loss, log_neptune

from sklearn.metrics import classification_report

def call_model(args, device):
    if args.encoder == 'rnn' and args.classifier == 'rnn':
        model = RNN_enc_RNN_clf(args)

    model = model.to(device)

    return model

def train_main(args, neptune):
    device = torch.device('cuda')

    model = call_model(args, device)

    if args.classifier == 'rnn':
        test_dnn = False
    else:
        test_dnn = True

    trainiter1 = AD_SUP2_RNN_ITERATOR2(mode='train',
                                      csv_path=args.csv_path1,
                                      ids_path=args.ids_path1,
                                      stat_path=args.stat_path1,
                                      dict_path=args.dict_path,
                                      data_name=args.data_name1,
                                      batch_size=args.batch_size,
                                      rnn_len=args.rnn_len,
                                      test_dnn=test_dnn,
                                      label=args.label)

    trainiter2 = AD_SUP2_RNN_ITERATOR2(mode='train',
                                       csv_path=args.csv_path2,
                                       ids_path=args.ids_path2,
                                       stat_path=args.stat_path2,
                                       dict_path=args.dict_path,
                                       data_name=args.data_name2,
                                       batch_size=args.batch_size,
                                       rnn_len=args.rnn_len,
                                       test_dnn=test_dnn,
                                       label=args.label)

    validiter1 = AD_SUP2_RNN_ITERATOR2(mode='valid',
                                       csv_path=args.csv_path1,
                                       ids_path=args.ids_path1,
                                       stat_path=args.stat_path1,
                                       dict_path=args.dict_path,
                                       data_name=args.data_name1,
                                       batch_size=args.batch_size,
                                       rnn_len=args.rnn_len,
                                       test_dnn=test_dnn,
                                       label=args.label)
    
    validiter2 = AD_SUP2_RNN_ITERATOR2(mode='valid',
                                       csv_path=args.csv_path2,
                                       ids_path=args.ids_path2,
                                       stat_path=args.stat_path2,
                                       dict_path=args.dict_path,
                                       data_name=args.data_name2,
                                       batch_size=args.batch_size,
                                       rnn_len=args.rnn_len,
                                       test_dnn=test_dnn,
                                       label=args.label)

    testiter1 = AD_SUP2_RNN_ITERATOR2(mode='test',
                                     csv_path=args.csv_path1,
                                     ids_path=args.ids_path1,
                                     stat_path=args.stat_path1,
                                     dict_path=args.dict_path,
                                     data_name=args.data_name1,
                                     batch_size=args.batch_size,
                                     rnn_len=args.rnn_len,
                                     test_dnn=test_dnn,
                                     label=args.label)

    testiter2 = AD_SUP2_RNN_ITERATOR2(mode='test',
                                     csv_path=args.csv_path2,
                                     ids_path=args.ids_path2,
                                     stat_path=args.stat_path2,
                                     dict_path=args.dict_path,
                                     data_name=args.data_name2,
                                     batch_size=args.batch_size,
                                     rnn_len=args.rnn_len,
                                     test_dnn=test_dnn,
                                     label=args.label)

    criterion = torch.nn.CrossEntropyLoss()

    print('trainiter1: {} samples'.format(len(trainiter1)))
    print('trainiter2: {} samples'.format(len(trainiter2)))
    print('validiter1: {} samples'.format(len(validiter1)))
    print('validiter2: {} samples'.format(len(validiter2)))
    print('testiter1: {} samples'.format(len(testiter1)))
    print('testiter2: {} samples'.format(len(testiter2)))

    # declare optimizer
    estring = "optim." + args.optimizer
    optimizer = eval(estring)(model.parameters(), lr=args.lr)
    if args.use_scheduler == 1:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # modify the dataset to produce labels
    # create a training loop
    train_loss = 0.0
    bc = 0 # bad counter
    sc = 0 # step counter
    best_valid_loss = None
    savedir = './result/' + args.out_file
    if args.label == 'rcl':
        with open(args.dict_path, 'rb') as fp:
            idx2class = (pkl.load(fp))['idx2class']

    for ei in range(args.max_epoch):
        for li, ((x_data1, y_data1, end_of_data1), (x_data2, y_data2, end_of_data2)) in enumerate(zip(trainiter1, trainiter2)):
            x_data1, x_data2 = x_data1.to(dtype=torch.float32, device=device), x_data2.to(dtype=torch.float32, device=device)
            y_data1, y_data2 = y_data1.to(dtype=torch.int64, device=device), y_data2.to(dtype=torch.int64, device=device)

            optimizer.zero_grad()

            output1 = model(x_data1)
            output2 = model(x_data2)

            # go through loss function
            loss1 = criterion(output1, y_data1)
            loss2 = criterion(output2, y_data2)
            combined_loss = loss1 + loss2
            combined_loss.backward()

            # optimizer
            optimizer.step()
            train_loss += combined_loss.item()

            if end_of_data2 == 1: break

        train_loss = train_loss / (li + 1)
        print('epoch: {:d} | train_loss: {:.4f}'.format(ei+1, train_loss))
        if neptune is not None: neptune.log_metric('train_loss', ei+1, train_loss)
        train_loss = 0.0

        valid_loss = get_combined_valid_loss(model, validiter1, validiter2, criterion, device)
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
                if args.use_scheduler == 1:
                    print("learning rate decay..")
                    scheduler.step()
                    bc = 0
                    sc += 1

                    if(sc >= args.n_decay):
                        break
                else:
                    print("early stopping..")
                    break

            print('bad counter == %d' % (bc))

    model = torch.load(savedir)
    
    # evaluation
    eval_modes = ['valid', 'test']
    iter_nums = ['1', '2']
    
    for iter_num in iter_nums:
        for eval_mode in eval_modes:
            dataiter_str = eval_mode + 'iter' + iter_num
            dataiter_name = eval('args.dataset' + iter_num)
            dataiter = eval(dataiter_str)

            targets, preds = eval_forward(model, dataiter, device)

            # metric
            acc, prec, rec, f1 = eval_binary(targets, preds)

            # std and neptune
            print('{} | acc: {:.4f} | prec: {:.4f} | rec: {:.4f} | f1: {:.4f} |'.format(eval_mode, acc, prec, rec, f1))
            if neptune is not None:
                neptune.set_property(dataiter_name + ' acc', acc)
                neptune.set_property(dataiter_name + ' prec', prec)
                neptune.set_property(dataiter_name + ' rec', rec)
                neptune.set_property(dataiter_name + ' f1', f1)

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
    parser.add_argument('--dataset1', type=str)
    parser.add_argument('--dataset2', type=str)
    parser.add_argument('--dim_input', type=int)
    parser.add_argument('--rnn_len', type=int)
    parser.add_argument('--csv_path1', type=str)
    parser.add_argument('--csv_path2', type=str)
    parser.add_argument('--ids_path1', type=str)
    parser.add_argument('--ids_path2', type=str)
    parser.add_argument('--stat_path1', type=str)
    parser.add_argument('--stat_path2', type=str)
    parser.add_argument('--dict_path', type=str)
    parser.add_argument('--data_name1', type=str)
    parser.add_argument('--data_name2', type=str)

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
    parser.add_argument('--drop_p', type=float)

    # learning rate decay
    parser.add_argument('--use_scheduler', type=int)
    parser.add_argument('--step_size', type=int)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--n_decay', type=int, default=3)

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