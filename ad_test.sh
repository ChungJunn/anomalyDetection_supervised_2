#!/bin/bash

EXP_NAME='20.1.17-testing'

MODEL=$2
TRAINED_DATASET=$3
MODEL_FILE=$4
MODEL_PATH='./result/'$MODEL_FILE'.pth'
REDUCE=$5
OPTIMIZER=$6
LR=$7
BATCH_SIZE=$8
DIM_LSTM_HIDDEN=$9
NLAYER=${10}
D_MODEL=${11}
NHEAD=${12}
DIM_FEEDFORWARD=${13}
TEST_DATASET=${14}

# check dataset and set csv paths
DATA_DIR=$HOME'/autoregressor/data/'$TRAINED_DATASET'_data/gnn_data/'
if [ $TRAINED_DATASET = 'cnsm_exp1' ]
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
        --model=$MODEL \
        --trained_dataset=$TRAINED_DATASET\
        --model_file=$MODEL_FILE\
        --reduce=$REDUCE\
        --optimizer=$OPTIMIZER\
        --lr=$LR\
        --batch_size=$BATCH_SIZE\
        --dim_lstm_hidden=$DIM_LSTM_HIDDEN\
        --nlayer=$NLAYER\
        --d_model=$D_MODEL\
        --nhead=$NHEAD\
        --dim_feedforward=$DIM_FEEDFORWARD\
        --test_dataset=$TEST_DATASET\
        --model_path=$MODEL_PATH \
        --data_dir=$DATA_DIR \
        --n_nodes=$N_NODES \
        --csv1=$CSV1 \
        --csv2=$CSV2 \
        --csv3=$CSV3 \
        --csv4=$CSV4 \
        --csv5=$CSV5 \
        --csv_label=$CSV_LABEL
