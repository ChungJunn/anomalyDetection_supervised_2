#!/bin/bash
DATASET=$1 #'cnsm_exp1, cnsm_exp2_1, or cnsm_exp2_2'

TUNE=1
PATIENCE=50 # not used by script
MAX_EPOCH=1000
N_SAMPLES=200
TEST_LOG_INTERVAL=5

# Model
ENCODER=$2 # rnn, transformer, none, bidirectionalrnn
if [ $ENCODER = 'bidirectionalrnn' ]
then
    BIDIRECTIONAL=1
else
    BIDIRECTIONAL=0
fi

# Simple model params
DIM_INPUT=22
USE_FEATURE_MAPPING=1

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

/usr/bin/python3.8 ad_main_tune.py  --data_dir=$DATA_DIR \
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
                        --bidirectional=$BIDIRECTIONAL \
                        --dim_input=$DIM_INPUT \
                        --n_samples=$N_SAMPLES \
                        --use_feature_mapping=$USE_FEATURE_MAPPING \
                        --encoder=$ENCODER \
                        --tune=$TUNE \
                        --test_log_interval=$TEST_LOG_INTERVAL
