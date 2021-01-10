#!/bin/bash
DATASET=$1 #'cnsm_exp1, cnsm_exp2_1, or cnsm_exp2_2'

PATIENCE=20
MAX_EPOCH=200

# Model
ENCODER='none' # rnn, transformer, none

# Simple model params
DIM_INPUT=22

# RNN params
BIDIRECTIONAL=1
DIM_LSTM_HIDDEN=64
DIM_LSTM_INPUT=22

# Transformer params
D_MODEL=22
NHEAD=2
DIM_FEEDFORWARD=128

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

python3 ad_main_tune.py  --data_dir=$DATA_DIR \
                        --csv1=$CSV1 \
                        --csv2=$CSV2 \
                        --csv3=$CSV3 \
                        --csv4=$CSV4 \
                        --csv5=$CSV5 \
                        --csv_label=$CSV_LABEL \
                        --n_nodes=$N_NODES \
                        --patience=$PATIENCE \
                        --dataset=$DATASET \
                        --max_epoch=$MAX_EPOCH \
                        --dim_lstm_hidden=$DIM_LSTM_HIDDEN \
                        --dim_lstm_input=$DIM_LSTM_INPUT \
                        --bidirectional=$BIDIRECTIONAL \
                        --d_model=$D_MODEL \
                        --nhead=$NHEAD \
                        --dim_feedforward=$DIM_FEEDFORWARD \
                        --dim_input=$DIM_INPUT \
                        --encoder=$ENCODER
#done
