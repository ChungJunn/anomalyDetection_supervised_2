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
from ad_data import AD_SUP2_ITERATOR
from ad_eval import eval_main

def validate(model, validiter, device, criterion):
    valid_loss = 0.0

    for li, (anno, label, end_of_data) in enumerate(validiter):
        anno = anno.to(dtype=torch.float32, device=device)
        label = label.to(dtype=torch.int64, device=device)

        # go through loss function
        output = model(anno)
        loss = criterion(output, label)

        # compute loss
        valid_loss += loss.item()
        if end_of_data == 1: break

    valid_loss /= (li+1)

    return valid_loss

def train_main(args, neptune):
    device = torch.device('cuda:0')
    criterion = F.nll_loss

    # declare model
    if args.encoder=='none':
        model = AD_SUP2_MODEL1(reduce=args.reduce, dim_input=args.dim_input).to(device)
    elif args.encoder=='rnn' or args.encoder=='bidirectionalrnn':
        model = AD_SUP2_MODEL2(dim_input=args.dim_input, dim_lstm_hidden=args.dim_lstm_hidden, reduce=args.reduce, bidirectional=args.bidirectional, use_feature_mapping=args.use_feature_mapping, dim_feature_mapping=args.dim_feature_mapping, nlayer=args.nlayer).to(device)
    elif args.encoder=='transformer':
        model = AD_SUP2_MODEL3(dim_input=args.dim_input, nhead=args.nhead, dim_feedforward=args.dim_feedforward, reduce=args.reduce, use_feature_mapping=args.use_feature_mapping, dim_feature_mapping=args.dim_feature_mapping, nlayer=args.nlayer).to(device)
    else:
        print("model must be either \'none\', \'rnn\', \'transformer\'")
        sys.exit(0)

    print('# model', model)

    csv_files=[]
    for n in range(1, args.n_nodes+1):
        csv_file = eval('args.csv' + str(n))
        csv_files.append(csv_file)
    csv_files.append(args.csv_label) # append label 

    # declare dataset
    trainiter = AD_SUP2_ITERATOR(tvt='sup_train', data_dir=args.data_dir, csv_files=csv_files, batch_size=args.batch_size)
    valiter = AD_SUP2_ITERATOR(tvt='sup_val', data_dir=args.data_dir, csv_files=csv_files, batch_size=args.batch_size)
    testiter = AD_SUP2_ITERATOR(tvt='sup_test', data_dir=args.data_dir, csv_files=csv_files, batch_size=args.batch_size)

    # declare optimizer
    estring = "optim." + args.optimizer
    optimizer = eval(estring)(model.parameters(), lr=args.lr)

    # modify the dataset to produce labels
    # create a training loop
    train_loss = 0.0
    log_interval=1000
    bc = 0
    best_val_f1 = None
    savedir = './result/' + args.out_file
    n_samples = trainiter.n_samples

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
            train_loss += loss.item()
        
            if li % log_interval == (log_interval - 1):
                train_loss = train_loss / log_interval
                print('epoch: {:d} | li: {:d} | train_loss: {:.4f}'.format(ei+1, li+1, train_loss))
                if neptune is not None: neptune.log_metric('train loss', (ei*n_samples)+(li+1), train_loss)
                train_loss = 0

            if end_of_data == 1: break

        # evaluation code
        # valid_loss = validate(model, valiter, device, criterion)
        acc,prec,rec,f1=eval_main(model,valiter,device,neptune=None)
        print('epoch: {:d} | valid_f1: {:.4f}'.format(ei+1, f1))
        if neptune is not None: neptune.log_metric('valid f1', ei, f1)

        # need to implement early-stop
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
    acc, prec, rec, f1 = eval_main(model, testiter, device, neptune=neptune)

    if neptune is not None:
        neptune.set_property('acc', acc)
        neptune.set_property('prec', prec)
        neptune.set_property('rec', rec)
        neptune.set_property('f1', f1)

    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--csv1', type=str)
    parser.add_argument('--csv2', type=str)
    parser.add_argument('--csv3', type=str)
    parser.add_argument('--csv4', type=str)
    parser.add_argument('--csv5', type=str)
    parser.add_argument('--csv_label', type=str)
    parser.add_argument('--n_nodes', type=int)
    parser.add_argument('--reduce', type=str)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--patience', type=float)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--max_epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--encoder', type=str)
    # Simple model params
    parser.add_argument('--dim_input', type=int)
    # RNN params
    parser.add_argument('--bidirectional', type=int)
    parser.add_argument('--dim_lstm_hidden', type=int)
    parser.add_argument('--use_feature_mapping', type=int)
    parser.add_argument('--dim_feature_mapping', type=int)
    # RNN and Transformer param
    parser.add_argument('--nlayer', type=int)
    # Transformer params 
    parser.add_argument('--nhead', type=int)
    parser.add_argument('--dim_feedforward', type=int)

    args = parser.parse_args()

    params = vars(args)

    neptune.init('cjlee/AnomalyDetection-GNN')
    experiment = neptune.create_experiment(name=args.exp_name, params=params)
    args.out_file = experiment.id + '.pth'

    print('parameters:')
    print('='*90)
    print(params)
    print('='*90)

    train_main(args, neptune)
