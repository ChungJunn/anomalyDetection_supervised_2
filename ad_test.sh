#!/bin/bash
EXP_NAME='20.1.17-testing'
MODEL_FILE=$2
MODEL_PATH='./result/'$MODEL_FILE'.pth'
DATASET=$3

# check dataset and set csv paths
DATA_DIR=$HOME'/autoregressor/data/'$DATASET'_data/gnn_data/'
if [ $DATASET = 'cnsm_exp1' ]
then
    CSV1='rnn_len16.fw.csv'
    CSV2='rnn_len16.ids.csv'
    CSV3='rnn_len16.flowmon.csv'
    CSV4='rnn_len16.dpi.csv'
    CSV5='rnn_len16.lb.csv'
    CSV_LABEL='rnn_len16.label.csv'
    
    N_NODES=5
else
    CSV1='rnn_len16.fw.csv'
    CSV2='rnn_len16.flowmon.csv'
    CSV3='rnn_len16.dpi.csv'
    CSV4='rnn_len16.ids.csv'
    CSV5=''
    CSV_LABEL='rnn_len16.label.csv'

    N_NODES=4
fi

export CUDA_VISIBLE_DEVICES=$1

python3 ad_test.py \
        --exp_name=$EXP_NAME \
        --model_path=$MODEL_PATH \
        --data_dir=$DATA_DIR \
        --n_nodes=$N_NODES \
        --csv1=$CSV1 \
        --csv2=$CSV2 \
        --csv3=$CSV3 \
        --csv4=$CSV4 \
        --csv5=$CSV5 \
        --csv_label=$CSV_LABEL
