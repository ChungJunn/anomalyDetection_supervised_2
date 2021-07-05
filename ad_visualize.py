import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os

from sklearn import preprocessing

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
    dataset = "cnsm_exp2_2"
    datadir = "../autoregressor/data/raw"
    data = pd.read_csv(datadir + "/" + dataset + "_data.csv")

    # split dataset to data and labels
    x_data = data.iloc[:, :-3]
    y_data = data.iloc[:, -3]

    n_samples, n_features = x_data.shape

    # y labels
    '''
    # remove '.' at the end
    y_list = list(y_data)
    y_list = [item.strip('.') for item in y_list]

    # encoder string labels to 0,1 labels
    encoder_dict = get_encoder_dict(y_list)
    y_list_encoded = []

    # convert to binary labels
    for i in range(len(y_list)):
        y_list_encoded.append(encoder_dict[y_list[i]])
    '''

    # given: x_data, y_data
    # output: plot#-#.png x 25
    len_fig_interval = 100000 
    n_subplots = 10 
    n_figs = (n_features // n_subplots) + 1 # number of figures for each fig_interval

    for iloop in range((n_samples // len_fig_interval) + 1): # how many figures horizontally? 
        fig_interval = range((len_fig_interval * iloop), min(n_samples, (len_fig_interval * (iloop + 1))))
        # get fig_sub_interval
        fig_sub_intervals = get_intervals(y_data[fig_interval[0]:fig_interval[-1]])

        for i in range(n_figs):
            fig, axs = plt.subplots(n_subplots)
            fig.set_figwidth(25.6)
            fig.set_figheight(16.8)
            
            for j in range(n_subplots):
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
                    
                    color = 'r' if label == 1 else 'c'
                    axs[j].plot(interval, x_data.iloc[interval, col_idx], c=color)
            
            savedir = "./plot_" + dataset
            if not os.path.exists(savedir):
                os.mkdir(savedir)

            savepath = savedir + "/plot{}-{}.png".format(iloop+1, i+1)
            fig.savefig(savepath)
            plt.clf()
            plt.close()
'''
if __name__ == '__main__':           
    # encode categorical variables
    cat_cols = ['protocol_type', 'service', 'flag']
    enc_dict = {}

    for item in cat_cols:
        le = preprocessing.LabelEncoder()
        x_data[item] = le.fit_transform(x_data[item])
        enc_dict[item] = le.classes_

    # select x axis and y axis 
    x_interval = 100000

    # make fig, subplot
    n_subplots = 10
    n_figs = (n_features // n_subplots) + 1 

    for iloop in range((n_samples // x_interval) + 1):
        interval = range((x_interval * iloop), min(n_samples, (x_interval * (iloop + 1))))

        for i in range(n_figs): 
            fig, axs = plt.subplots(n_subplots)
            fig.set_figwidth(12.8)
            fig.set_figheight(16.8)
            for j in range(n_subplots):
                col_idx = min(i * n_subplots + j, n_features-1)
                axs[j].plot(interval, x_data.iloc[interval, col_idx]) 
                axs[j].set_title(x_data.columns[col_idx])
                if j != (n_subplots-1):
                    axs[j].xaxis.set_visible(False)

        # save figs
            plt.xlim(interval[0], interval[-1])
            fig.savefig('./plot{}-{}.png'.format(iloop+1, i+1))
            plt.clf()
            plt.close()
'''