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

from ad_model import AD_SUP2_MODEL1
from ad_model import AD_SUP2_MODEL2
from ad_model import AD_SUP2_MODEL3
from ad_model import AD_SUP2_MODEL4
from ad_model import AD_SUP2_MODEL5
from ad_data import AD_SUP2_ITERATOR, AD_SUP2_RNN_ITERATOR, AD_SUP2_DNN_ITERATOR
from ad_eval import eval_main
from ad_test import test

from ray import tune

def train_main(args, neptune):
    device = torch.device('cuda:0')
    criterion = F.nll_loss

    # declare model
    if args.encoder=='none':
        model = AD_SUP2_MODEL1(reduce=args.reduce, dim_input=args.dim_input).to(device)
    elif args.encoder=='rnn' or args.encoder=='bidirectionalrnn':
        model = AD_SUP2_MODEL2(dim_input=args.dim_input, dim_lstm_hidden=args.dim_lstm_hidden, reduce=args.reduce, bidirectional=args.bidirectional, use_feature_mapping=args.use_feature_mapping, dim_feature_mapping=args.dim_feature_mapping, nlayer=args.nlayer,dim_att=args.dim_att).to(device)
    elif args.encoder=='transformer':
        model = AD_SUP2_MODEL3(dim_input=args.dim_input, nhead=args.nhead, dim_feedforward=args.dim_feedforward, reduce=args.reduce, use_feature_mapping=args.use_feature_mapping, dim_feature_mapping=args.dim_feature_mapping, nlayer=args.nlayer).to(device)
    elif args.encoder=='dnn':
        model = AD_SUP2_MODEL5(dim_input=args.dim_input, dim_enc=args.dim_enc, reduce=args.reduce).to(device)
    elif args.encoder=='rnn-classifier':
        model = AD_SUP2_MODEL4(dim_input=args.dim_input, dim_lstm_hidden=args.dim_lstm_hidden, reduce=args.reduce, bidirectional=args.bidirectional, use_feature_mapping=args.use_feature_mapping, dim_feature_mapping=args.dim_feature_mapping, nlayer=args.nlayer, dim_att=args.dim_att, clf_dim_lstm_hidden=args.clf_dim_lstm_hidden, clf_dim_fc_hidden=args.clf_dim_fc_hidden, clf_dim_output=args.clf_dim_output).to(device)
    else:
        print("model must be either \'none\',\'dnn\', \'rnn\', \'transformer\'")
        sys.exit(0)

    print('# model', model)

    pkl_files=[]
    for n in range(1, args.n_nodes+1):
        pkl_file = eval('args.pkl_file' + str(n))
        pkl_files.append(pkl_file)
    pkl_files.append(args.pkl_label) # append label 

    # declare dataset
    '''
    trainiter = AD_SUP2_ITERATOR(tvt='sup_train', data_dir=args.data_dir, pkl_files=pkl_files, batch_size=args.batch_size)
    valiter = AD_SUP2_ITERATOR(tvt='sup_val', data_dir=args.data_dir, pkl_files=pkl_files, batch_size=args.batch_size)
    testiter = AD_SUP2_ITERATOR(tvt='sup_test', data_dir=args.data_dir, pkl_files=pkl_files, batch_size=args.batch_size)
    '''

    # declare dataset
    trainiter = AD_SUP2_DNN_ITERATOR(tvt='sup_train', data_dir=args.data_dir, pkl_files=pkl_files, batch_size=args.batch_size)
    valiter = AD_SUP2_DNN_ITERATOR(tvt='sup_val', data_dir=args.data_dir, pkl_files=pkl_files, batch_size=args.batch_size)
    testiter = AD_SUP2_DNN_ITERATOR(tvt='sup_test', data_dir=args.data_dir, pkl_files=pkl_files, batch_size=args.batch_size)

    '''
    # declare dataset
    trainiter = AD_SUP2_RNN_ITERATOR(tvt='sup_train', data_dir=args.data_dir, pkl_files=pkl_files, batch_size=args.batch_size)
    valiter = AD_SUP2_RNN_ITERATOR(tvt='sup_val', data_dir=args.data_dir, pkl_files=pkl_files, batch_size=args.batch_size)
    testiter = AD_SUP2_RNN_ITERATOR(tvt='sup_test', data_dir=args.data_dir, pkl_files=pkl_files, batch_size=args.batch_size)
    '''

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
    if args.tune == 0:
        savedir = './result/' + args.out_file
    n_samples = trainiter.n_samples
    clf_hidden = None

    for ei in range(args.max_epoch):
        for li, (anno, label, end_of_data) in enumerate(trainiter):
            anno = anno.to(dtype=torch.float32, device=device)
            label = label.to(dtype=torch.int64, device=device)

            optimizer.zero_grad()

            # output, clf_hidden = model(anno, clf_hidden)
            # clf_hidden = [clf_hidden[0].detach(), clf_hidden[1].detach()]

            output = model(anno)

            # go through loss function
            loss = criterion(output, label)
            loss.backward()

            # optimizer
            optimizer.step()
            #train_loss1 += loss.item()
            train_loss2 += loss.item()

            if end_of_data == 1: break

            '''
            if (log_idx % log_interval) == (log_interval - 1):
                print('{:d} | {:d} | {:.4f}'.format(ei+1, log_idx+1, train_loss1))
                if neptune is not None: neptune.log_metric('train loss (n_samples)', log_idx+1, train_loss1)
                train_loss1 = 0.0
            log_idx += 1
            '''

        train_loss = train_loss2 / li
        print('epoch: {:d} | train_loss: {:.4f}'.format(ei+1, train_loss2))
        if neptune is not None: neptune.log_metric('train loss2', ei, train_loss2)
        train_loss2 = 0.0

        acc,prec,rec,f1=eval_main(model,trainiter,device,neptune=None)
        print('epoch: {:d} | train_f1: {:.4f}'.format(ei+1, f1))
        if neptune is not None: neptune.log_metric('train f1', ei, f1)
        # evaluation code
        '''
        acc,prec,rec,f1=eval_main(model,valiter,device,neptune=None)
        print('epoch: {:d} | valid_f1: {:.4f}'.format(ei+1, f1))
        if neptune is not None: neptune.log_metric('valid f1', ei, f1)
        '''

        # if tune, report the metrics / also test metric every 5 epochs
        if args.tune == 1:
            tune.report(train_loss=train_loss, val_f1=f1, test_f1=-1)
            if ei % args.test_log_interval == (args.test_log_interval-1):
                test_acc,test_prec,test_rec,test_f1=eval_main(model,testiter,device,neptune=None)
                tune.report(train_loss=train_loss, val_f1=f1, test_f1=test_f1)

        if ei == 0 or f1 > best_val_f1:
            if args.tune == 0:
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

    if args.tune == 0:
        model = torch.load(savedir)
        
        datasets = ['cnsm_exp1', 'cnsm_exp2_1', 'cnsm_exp2_2']
        for dset in datasets:
            acc, prec, rec, f1 = test(model, dset, args.batch_size, device, neptune)

            if neptune is not None:
                neptune.set_property(dset+'_acc', acc)
                neptune.set_property(dset+'_prec', prec)
                neptune.set_property(dset+'_rec', rec)
                neptune.set_property(dset+'_f1', f1)

    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--pkl_file1', type=str)
    parser.add_argument('--pkl_file2', type=str)
    parser.add_argument('--pkl_file3', type=str)
    parser.add_argument('--pkl_file4', type=str)
    parser.add_argument('--pkl_file5', type=str)
    parser.add_argument('--pkl_label', type=str)
    parser.add_argument('--n_nodes', type=int)
    parser.add_argument('--reduce', type=str)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--patience', type=float)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--max_epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--encoder', type=str)
    parser.add_argument('--tune', type=int)
    # Simple model params
    parser.add_argument('--dim_input', type=int)
    # RNN params
    parser.add_argument('--bidirectional', type=int)
    parser.add_argument('--dim_lstm_hidden', type=int)
    parser.add_argument('--dim_att', type=int)
    # RNN and Transformer param
    parser.add_argument('--nlayer', type=int)
    parser.add_argument('--use_feature_mapping', type=int)
    parser.add_argument('--dim_feature_mapping', type=int)
    # Transformer params 
    parser.add_argument('--nhead', type=int)
    parser.add_argument('--dim_feedforward', type=int)
    # DNN-enc params
    parser.add_argument('--dim_enc', type=int)
    # clf params
    parser.add_argument('--clf_dim_lstm_hidden', type=int)
    parser.add_argument('--clf_dim_fc_hidden', type=int)
    parser.add_argument('--clf_dim_output', type=int)

    args = parser.parse_args()

    params = vars(args)

    #neptune.init('cjlee/AnomalyDetection-Supervised')
    #experiment = neptune.create_experiment(name=args.exp_name, params=params)
    #args.out_file = experiment.id + '.pth'

    neptune=None
    args.out_file = 'dummy.pth'

    print('parameters:')
    print('='*90)
    print(params)
    print('='*90)

    train_main(args, neptune)
