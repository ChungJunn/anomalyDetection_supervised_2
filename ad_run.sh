#!/bin/bash
EXP_NAME="210713.use_label_information"

# task
LABEL='sla'
USE_NEPTUNE=1

# dataset
DATASET=$2 #cnsm_exp1, cnsm_exp2_1, or cnsm_exp2_2'

if [ $DATASET == 'cnsm_exp1' ] || [ $DATASET == 'cnsm_exp2_1' ] || [ $DATASET == 'cnsm_exp2_2' ]
then
    DIM_INPUT=24 # added 1 for label information
elif [ $DATASET == 'tpi_train' ]
then
    DIM_INPUT=6
else
    echo '$DATASET must be either cnsm_exp1, cnsm_exp2_1, or cnsm_exp2_2'
    exit -1
fi
RNN_LEN=16

## dataset
BASE_DIR=$HOME'/autoregressor/data/'
CSV_PATH=$BASE_DIR'raw/'$DATASET'_data.csv'
IDS_PATH=$BASE_DIR''$DATASET'_data/indices.rnn_len16.pkl'
DICT_PATH=$BASE_DIR''$DATASET'_data/dict.pkl'
STAT_PATH=$CSV_PATH'.stat'
DATA_NAME=$DATASET'_data'

# teacher-focring
TEACHER_FORCING_RATIO=$4
# fm
DIM_FEATURE_MAPPING=24
# enc
ENCODER="transformer"
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
REDUCE="self-attention" # mean, max, or self-attention

# clf
CLASSIFIER=$3 # dnn or rnn
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
                        --csv_path=$CSV_PATH \
                        --ids_path=$IDS_PATH \
                        --stat_path=$STAT_PATH \
                        --data_name=$DATA_NAME \
                        --rnn_len=$RNN_LEN \
                        --label=$LABEL \
                        --dict_path=$DICT_PATH \
                        --use_neptune=$USE_NEPTUNE \
                        --use_scheduler=$USE_SCHEDULER \
                        --step_size=$STEP_SIZE \
                        --gamma=$GAMMA \
                        --n_decay=$N_DECAY \
                        --drop_p=$DROP_P \
                        --teacher_forcing_ratio=$TEACHER_FORCING_RATIO
done
exit 0