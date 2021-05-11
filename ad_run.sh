#!/bin/bash
EXP_NAME='210511.debug'

# task
LABEL='rcl'

# dataset
DATASET=$2 #'cnsm_exp1, cnsm_exp2_1, or cnsm_exp2_2'
DIM_INPUT=23
RNN_LEN=16
## dataset
BASE_DIR=$HOME'/autoregressor/data/'
CSV_PATH=$BASE_DIR'raw/'$DATASET'_data.csv'
IDS_PATH=$BASE_DIR''$DATASET'_data/indices.rnn_len16.pkl'
DICT_PATH=$BASE_DIR''$DATASET'_data/dict.pkl'
STAT_PATH=$CSV_PATH'.stat'
DATA_NAME=$DATASET'_data'

# fm
USE_FEATURE_MAPPING=1
DIM_FEATURE_MAPPING=24

# enc
ENCODER='transformer'
NLAYER=2
## DNN-enc
DIM_ENC=-1
## RNN-enc
BIDIRECTIONAL=-1
DIM_LSTM_HIDDEN=-1
## transformer-enc
NHEAD=4
DIM_FEEDFORWARD=48

# readout
REDUCE='max' # mean, max, or self-attention
DIM_ATT=-1

# clf
CLASSIFIER='dnn' # dnn or rnn
CLF_N_LSTM_LAYERS=-1
CLF_N_FC_LAYERS=3
CLF_DIM_LSTM_HIDDEN=-1
CLF_DIM_FC_HIDDEN=600

if [ $LABEL == 'sla' ]
then
    CLF_DIM_OUTPUT=2
elif [ $LABEL == 'rcl' ]
then
    CLF_DIM_OUTPUT=7
else
    echo '$LABEL must be either sla or rcl'
    exit -1
fi

# training parameter
OPTIMIZER='Adam'
LR=0.001
BATCH_SIZE=64
PATIENCE=20
MAX_EPOCH=1000

export CUDA_VISIBLE_DEVICES=$1
#for i in 1 2 3
#do
    /usr/bin/python3.8 ad_main.py \
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
                        --dim_enc=$DIM_ENC \
                        --clf_n_lstm_layers=$CLF_N_LSTM_LAYERS \
                        --clf_n_fc_layers=$CLF_N_FC_LAYERS \
                        --clf_dim_lstm_hidden=$CLF_DIM_LSTM_HIDDEN \
                        --clf_dim_fc_hidden=$CLF_DIM_FC_HIDDEN \
                        --clf_dim_output=$CLF_DIM_OUTPUT \
                        --csv_path=$CSV_PATH \
                        --ids_path=$IDS_PATH \
                        --stat_path=$STAT_PATH \
                        --data_name=$DATA_NAME \
                        --rnn_len=$RNN_LEN \
                        --label=$LABEL \
                        --dict_path=$DICT_PATH
#done

exit 0
