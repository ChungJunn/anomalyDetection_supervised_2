#!/bin/bash
EXP_NAME='21.02.16.exp9'

# Weight for combined_loss
ALPHA=0.5

# Model
ENCODER=$2 # rnn, transformer, none
BIDIRECTIONAL=$3

BATCH_SIZE=64
DIM_ENC=-1
DIM_LSTM_HIDDEN=$4
DIM_ATT=$DIM_LSTM_HIDDEN

USE_FEATURE_MAPPING=1
DIM_FEATURE_MAPPING=24

NLAYER=2
OPTIMIZER='Adam'
LR=0.001
REDUCE='self-attention' # mean, max, or last_hidden
NHEAD=4
DIM_FEEDFORWARD=48

# other fixed params
PATIENCE=20
MAX_EPOCH=1000
DIM_INPUT=22

# check dataset and set csv paths
DATA_DIR=$HOME'/autoregressor/data/cnsm_exp1_data/gnn_data/'
DATA_DIR2=$HOME'/autoregressor/data/cnsm_exp2_2_data/gnn_data/'

CSV1='rnn_len16.fw.csv'
CSV2='rnn_len16.ids.csv'
CSV3='rnn_len16.flowmon.csv'
CSV4='rnn_len16.dpi.csv'
CSV5='rnn_len16.lb.csv'
CSV_LABEL='rnn_len16.label.csv'
    
N_NODES=5

export CUDA_VISIBLE_DEVICES=$1

for i in 1 2 3 4 5 
do
/usr/bin/python3.8 ad_dual_main.py  --data_dir=$DATA_DIR \
                    --data_dir2=$DATA_DIR2 \
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
                    --alpha=$ALPHA \
                    --dim_enc=$DIM_ENC
done
