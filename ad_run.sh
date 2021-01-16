#!/bin/bash
EXP_NAME='21.01.15.exp1'

# Model
ENCODER=$2 # rnn, transformer, none
DATASET=$3 #'cnsm_exp1, cnsm_exp2_1, or cnsm_exp2_2'
BATCH_SIZE=$4
REDUCE=$8 # mean, max, or last_hidden
OPTIMIZER=$7
LR=$6
PATIENCE=20
MAX_EPOCH=2000

# Simple model params
DIM_INPUT=22

# RNN params
BIDIRECTIONAL=1
DIM_LSTM_HIDDEN=$5
DIM_LSTM_INPUT=22

# Transformer params
D_MODEL=22
NHEAD=$9
DIM_FEEDFORWARD=${10}

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

for i in 1 2 3 4 5
do
    python3 ad_main.py  --data_dir=$DATA_DIR \
                        --csv1=$CSV1 \
                        --csv2=$CSV2 \
                        --csv3=$CSV3 \
                        --csv4=$CSV4 \
                        --csv5=$CSV5 \
                        --csv_label=$CSV_LABEL \
                        --n_nodes=$N_NODES \
                        --reduce=$REDUCE \
                        --optimizer=$OPTIMIZER \
                        --lr=$LR \
                        --patience=$PATIENCE \
                        --exp_name=$EXP_NAME \
                        --dataset=$DATASET \
                        --max_epoch=$MAX_EPOCH \
                        --batch_size=$BATCH_SIZE \
                        --dim_lstm_hidden=$DIM_LSTM_HIDDEN \
                        --dim_lstm_input=$DIM_LSTM_INPUT \
                        --bidirectional=$BIDIRECTIONAL \
                        --d_model=$D_MODEL \
                        --nhead=$NHEAD \
                        --dim_feedforward=$DIM_FEEDFORWARD \
                        --dim_input=$DIM_INPUT \
                        --encoder=$ENCODER
done
