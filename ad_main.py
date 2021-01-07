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
    model = AD_SUP2_MODEL1(reduce=args.reduce).to(device)

    csv_files=[]
    for n in range(1, args.n_nodes+1):
        csv_file = eval('args.csv' + str(n))
        csv_files.append(csv_file)
    csv_files.append(args.csv_label) # append label 

    # declare dataset
    trainiter = AD_SUP2_ITERATOR(tvt='sup_train', data_dir=args.data_dir, csv_files=csv_files)
    valiter = AD_SUP2_ITERATOR(tvt='sup_val', data_dir=args.data_dir, csv_files=csv_files)
    testiter = AD_SUP2_ITERATOR(tvt='sup_test', data_dir=args.data_dir, csv_files=csv_files)

    # declare optimizer
    estring = "optim." + args.optimizer
    optimizer = eval(estring)(model.parameters(), lr=args.lr)

    # modify the dataset to produce labels
    # create a training loop
    train_loss = 0.0
    train_loss2 = 0.0
    log_interval=1000
    bc = 0
    best_val = None
    savedir = './result/' + args.out_file
    n_samples = trainiter.n_samples

    for ei in range(1000):
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
        valid_loss = validate(model, valiter, device, criterion)
        print('epoch: {:d} | valid_loss: {:.4f}'.format(ei+1, valid_loss))
        if neptune is not None: neptune.log_metric('valid loss', ei, valid_loss)

        # need to implement early-stop
        if ei == 0 or valid_loss < best_val:
            torch.save(model, savedir)
            bc = 0
            best_val = valid_loss
            print('found new best model')
        else:
            bc += 1
            if bc > args.patience:
                print('early stopping..')
                break
            print('bad counter == %d' % (bc))

    model = torch.load(savedir)
    acc, prec, rec, f1 = eval_main(model, testiter, device, neptune=neptune)

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
