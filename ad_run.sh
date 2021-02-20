#!/bin/bash
EXP_NAME=$2
TUNE=0

# Model
ENCODER=$3 # rnn, transformer, none, dnn
BIDIRECTIONAL=$4

DATASET=$5 #'cnsm_exp1, cnsm_exp2_1, or cnsm_exp2_2'
BATCH_SIZE=64
DIM_LSTM_HIDDEN=$6
DIM_ATT=$DIM_LSTM_HIDDEN

USE_FEATURE_MAPPING=1
DIM_FEATURE_MAPPING=24

NLAYER=2
OPTIMIZER='Adam'
LR=0.001
REDUCE=$7 # mean, max, or self-attention
NHEAD=4
DIM_FEEDFORWARD=48

# DNN-enc params
DIM_ENC=24

# other fixed params
PATIENCE=20
MAX_EPOCH=1000
DIM_INPUT=22

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
    /usr/bin/python3.8 ad_main.py  --data_dir=$DATA_DIR \
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
                        --use_feature_mapping=$USE_FEATURE_MAPPING \
                        --dim_feature_mapping=$DIM_FEATURE_MAPPING \
                        --nlayer=$NLAYER \
                        --bidirectional=$BIDIRECTIONAL \
                        --nhead=$NHEAD \
                        --dim_feedforward=$DIM_FEEDFORWARD \
                        --dim_input=$DIM_INPUT \
                        --encoder=$ENCODER \
                        --dim_att=$DIM_ATT \
                        --tune=$TUNE \
                        --dim_enc=$DIM_ENC
done
