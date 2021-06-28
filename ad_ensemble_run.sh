#!/bin/bash
EXP_NAME="210619_rnn_clf_ensemble"

# task
LABEL='sla'

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
DIM_FEATURE_MAPPING=24

# enc
ENCODER='rnn'
NLAYER=2
## DNN-enc
DIM_ENC=-1
## RNN-enc
BIDIRECTIONAL=1
DIM_LSTM_HIDDEN=20
## transformer-enc
NHEAD=-1
DIM_FEEDFORWARD=-1

# readout
REDUCE='mean' # mean, max, or self-attention
DIM_ATT=-1

# clf
CLASSIFIER='rnn' # dnn or rnn
CLF_N_LSTM_LAYERS=1
CLF_N_FC_LAYERS=3
CLF_DIM_LSTM_HIDDEN=200
CLF_DIM_FC_HIDDEN=600

if [ $LABEL == 'sla' ]
then
    CLF_DIM_OUTPUT=2
else
    echo '$LABEL must be sla'
    exit -1
fi

# training parameter
OPTIMIZER='Adam'
LR=0.001
DROP_P=0.0
BATCH_SIZE=64
PATIENCE=10

# ensemble parameters
N_ESTIMATORS=$3

export CUDA_VISIBLE_DEVICES=$1
    /usr/bin/python3.8 ad_ensemble_train.py \
                        --reduce=$REDUCE \
                        --optimizer=$OPTIMIZER \
                        --lr=$LR \
                        --patience=$PATIENCE \
                        --exp_name=$EXP_NAME \
                        --dataset=$DATASET \
                        --batch_size=$BATCH_SIZE \
                        --dim_lstm_hidden=$DIM_LSTM_HIDDEN \
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
                        --dict_path=$DICT_PATH \
                        --n_estimators=$N_ESTIMATORS \
                        --drop_p=$DROP_P
                        
exit 0
