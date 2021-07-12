import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import sys

from ad_utils import call_model
from ad_data import AD_RNN_Dataset
from sklearn import preprocessing
from ad_test import eval_forward
import torch
import argparse

def get_intervals(labels):
    # initialize
    n_samples = len(labels)
    intervals = []
    start = 0
    idx = 1

    while(1):
        if idx >= n_samples:
            my_tuple = ((start, idx), labels[idx-1])
            intervals.append(my_tuple)
            break
        
        elif labels[idx] != labels[idx-1]:
            my_tuple = ((start, idx), labels[idx-1])
            intervals.append(my_tuple)
            start = idx
        
        idx += 1
    
    return intervals

def get_encoder_dict(labels):
    unique_list = list(set(labels))
    encoder_dict = {'normal':1}

    for item in unique_list:
        if item != 'normal':
            encoder_dict[item] = 0

    return encoder_dict

if __name__ == '__main__':
    # feature mapping
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_feature_mapping', type=int)
    parser.add_argument('--analyze', type=str)

    # dataset
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dim_input', type=int)
    parser.add_argument('--rnn_len', type=int)
    parser.add_argument('--csv_path', type=str)
    parser.add_argument('--ids_path', type=str)
    parser.add_argument('--stat_path', type=str)
    parser.add_argument('--dict_path', type=str)
    parser.add_argument('--data_name', type=str)

    # load_path
    parser.add_argument('--load_path', type=str)

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

    # clf
    parser.add_argument('--classifier', type=str)
    parser.add_argument('--clf_n_lstm_layers', type=int)
    parser.add_argument('--clf_n_fc_layers', type=int)
    parser.add_argument('--clf_dim_lstm_hidden', type=int)
    parser.add_argument('--clf_dim_fc_hidden', type=int)
    parser.add_argument('--clf_dim_output', type=int)

    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--drop_p', type=float)
    args = parser.parse_args()

    if args.classifier == "rnn":
        test_dnn = False
    else:
        test_dnn = True
    
    # obtain prediction from model
    device = torch.device("cuda")

    # obtain whether prediction is correct
    iter_data = AD_RNN_Dataset(mode="plot",
                            csv_path=args.csv_path,
                            ids_path=args.ids_path,
                            stat_path=args.stat_path,
                            data_name=args.data_name,
                            rnn_len=args.rnn_len,
                            test_dnn=test_dnn)
    
    dataiter = torch.utils.data.DataLoader(iter_data, batch_size=args.batch_size, shuffle=False)
    # model = call_model(args, device)
    # call parameters
    model = torch.load(args.load_path)
    
    y_target, y_pred = eval_forward(model, dataiter, device)
    y_target = y_target.squeeze()
    is_correct = (y_target == y_pred)

    dataset = args.dataset
    datadir = "../autoregressor/data/raw"
    data = pd.read_csv(datadir + "/" + dataset + "_data.csv")

    # split dataset to data and labels
    x_data = data.iloc[:, :-3]
    y_data = np.array(data.iloc[:, -3]).astype(np.int64)

    n_samples, n_features = x_data.shape
    # given: x_data, y_data
    # output: plot#-#.png x 25
    len_fig_interval = 5000
    n_subplots = 10 
    n_figs = (n_features // n_subplots) + 1 # number of figures for each fig_interval

    if args.analyze == "abnormal":
        anal_label = 1
    elif args.analyze == "normal":
        anal_label = 0
    else:
        print("args.analyze must be either normal or abnormal")

    for iloop in range((n_samples // len_fig_interval) + 1): # how many figures horizontally? 
        fig_interval = range((len_fig_interval * iloop), min(n_samples, (len_fig_interval * (iloop + 1))))
        # get fig_sub_interval
        fig_sub_intervals = get_intervals(y_data[fig_interval[0]:fig_interval[-1]])
        # obtain the tuple of correct samples

        for i in range(n_figs):
            print("{:d}th figure processing..".format(i+1))
            fig, axs = plt.subplots(n_subplots)
            fig.set_figwidth(25.6)
            fig.set_figheight(16.8)
            
            for j in range(n_subplots):
                print("{:d}th subplot processing..".format(j+1))
                col_idx = i * n_subplots + j
                if col_idx > n_features-1:
                    break

                axs[j].set_title(x_data.columns[col_idx])
                axs[j].set_xlim(fig_interval[0], fig_interval[-1])
                # plt.xlim(fig_interval[0], fig_interval[-1])
                
                if j != (n_subplots-1):
                    axs[j].xaxis.set_visible(False)

                for k in range(len(fig_sub_intervals)):
                    base_idx = len_fig_interval * iloop
                    start = fig_sub_intervals[k][0][0] + base_idx
                    end = fig_sub_intervals[k][0][1] + base_idx

                    interval = range(start, end)
                    label = fig_sub_intervals[k][1]
                    
                    if label != anal_label:
                        color = 'k'
                        axs[j].scatter(interval, x_data.iloc[interval, col_idx], c=color, s=1)
                    else:
                        for kk in range(start, end):
                            if kk >= len(is_correct):
                                continue
                            color = "blue" if is_correct[kk] == 1 else "red"
                            axs[j].scatter(kk, x_data.iloc[kk, col_idx], c=color, s=1)
                    
            savedir = "./plot_" + dataset + "_" + args.classifier + "_" + args.analyze
            if not os.path.exists(savedir):
                os.mkdir(savedir)

            savepath = savedir + "/plot{}-{}.png".format(iloop+1, i+1)
            fig.savefig(savepath)
            plt.clf()
            plt.close()
