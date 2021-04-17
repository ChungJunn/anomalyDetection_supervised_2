#!/bin/bash
EXP_NAME='21.04.14.exp1'
TUNE=0

# Model
ENCODER=$2 # currently not used
CLASSIFIER=$3 # dnn or rnn
BIDIRECTIONAL=0
RNN_LEN=16

DATASET=$4 #'cnsm_exp1, cnsm_exp2_1, or cnsm_exp2_2'
BATCH_SIZE=1
DIM_LSTM_HIDDEN=64
DIM_ATT=$DIM_LSTM_HIDDEN

USE_FEATURE_MAPPING=1
DIM_FEATURE_MAPPING=24

NLAYER=2
OPTIMIZER='Adam'
LR=0.001
REDUCE=$5 # mean, max, or self-attention
NHEAD=4
DIM_FEEDFORWARD=48

# DNN-enc params
DIM_ENC=24

# CLF params
CLF_DIM_LSTM_HIDDEN=64
CLF_DIM_FC_HIDDEN=200
CLF_DIM_OUTPUT=2

# other fixed params
PATIENCE=20
MAX_EPOCH=1000
DIM_INPUT=23

BASE_DIR=$HOME'/autoregressor/data/'

CSV_PATH=$BASE_DIR'raw/'$DATASET'_data.csv'
IDS_PATH=$BASE_DIR''$DATASET'_data/indices.rnn_len16.pkl'
STAT_PATH=$CSV_PATH'.stat'
DATA_NAME=$DATASET'_data'
RNN_LEN=16

export CUDA_VISIBLE_DEVICES=$1
#for i in 1 2 3 4 5
#do
    /usr/bin/python3.8 ad_main.py  --data_dir=$DATA_DIR \
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
                        --classifier=$CLASSIFIER \
                        --dim_att=$DIM_ATT \
                        --tune=$TUNE \
                        --dim_enc=$DIM_ENC \
                        --clf_dim_lstm_hidden=$CLF_DIM_LSTM_HIDDEN \
                        --clf_dim_fc_hidden=$CLF_DIM_FC_HIDDEN \
                        --clf_dim_output=$CLF_DIM_OUTPUT \
                        --csv_path=$CSV_PATH \
                        --ids_path=$IDS_PATH \
                        --stat_path=$STAT_PATH \
                        --data_name=$DATA_NAME \
                        --rnn_len=$RNN_LEN
#done
