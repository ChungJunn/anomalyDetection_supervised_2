#!/bin/bash
EXP_NAME='210701.ni_meeting_jointTraining'

# task
LABEL='sla'
USE_NEPTUNE=1

# dataset
DIM_INPUT=23
RNN_LEN=16

DATASET1='cnsm_exp1'
DATASET2='cnsm_exp2_2'

## dataset
BASE_DIR=$HOME'/autoregressor/data/'
CSV_PATH1=$BASE_DIR'raw/'$DATASET1'_data.csv'
CSV_PATH2=$BASE_DIR'raw/'$DATASET2'_data.csv'

IDS_PATH1=$BASE_DIR''$DATASET1'_data/indices.rnn_len16.pkl'
IDS_PATH2=$BASE_DIR''$DATASET2'_data/indices.rnn_len16.pkl'

DICT_PATH=$BASE_DIR''$DATASET1'_data/dict.pkl'

STAT_PATH1=$CSV_PATH1'.stat'
STAT_PATH2=$CSV_PATH2'.stat'

DATA_NAME1=$DATASET1'_data'
DATA_NAME2=$DATASET2'_data'

# fm
DIM_FEATURE_MAPPING=24

# enc
ENCODER=$2
NLAYER=2
## DNN-enc
DIM_ENC=-1
## RNN-enc
BIDIRECTIONAL=1
DIM_LSTM_HIDDEN=20
## transformer-enc
NHEAD=4
DIM_FEEDFORWARD=48

# readout
REDUCE=$3 # mean, max, or self-attention

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
    echo '$LABEL must be either sla'
    exit -1
fi

# training parameter
OPTIMIZER='Adam'
LR=0.001
DROP_P=0.0
BATCH_SIZE=64
PATIENCE=10
MAX_EPOCH=1000

USE_SCHEDULER=0
STEP_SIZE=1
GAMMA=0.5
N_DECAY=3

export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZDBmMTBmOS0zZDJjLTRkM2MtOTA0MC03YmQ5OThlZTc5N2YifQ=="
export CUDA_VISIBLE_DEVICES=$1

for i in 1 2 3
do
    /usr/bin/python3.8 ad_joint_train.py \
                        --reduce=$REDUCE \
                        --optimizer=$OPTIMIZER \
                        --lr=$LR \
                        --patience=$PATIENCE \
                        --exp_name=$EXP_NAME \
                        --dataset1=$DATASET1 \
                        --dataset2=$DATASET2 \
                        --max_epoch=$MAX_EPOCH \
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
                        --dim_enc=$DIM_ENC \
                        --clf_n_lstm_layers=$CLF_N_LSTM_LAYERS \
                        --clf_n_fc_layers=$CLF_N_FC_LAYERS \
                        --clf_dim_lstm_hidden=$CLF_DIM_LSTM_HIDDEN \
                        --clf_dim_fc_hidden=$CLF_DIM_FC_HIDDEN \
                        --clf_dim_output=$CLF_DIM_OUTPUT \
                        --csv_path1=$CSV_PATH1 \
                        --csv_path2=$CSV_PATH2 \
                        --ids_path1=$IDS_PATH1 \
                        --ids_path2=$IDS_PATH2 \
                        --stat_path1=$STAT_PATH1 \
                        --stat_path2=$STAT_PATH2 \
                        --data_name1=$DATA_NAME1 \
                        --data_name2=$DATA_NAME2 \
                        --rnn_len=$RNN_LEN \
                        --label=$LABEL \
                        --dict_path=$DICT_PATH \
                        --use_neptune=$USE_NEPTUNE \
                        --use_scheduler=$USE_SCHEDULER \
                        --step_size=$STEP_SIZE \
                        --gamma=$GAMMA \
                        --n_decay=$N_DECAY \
                        --drop_p=$DROP_P
done

exit 0
