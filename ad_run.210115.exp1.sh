#!/bin/bash

DATA1='cnsm_exp1'
DATA2='cnsm_exp2_1'
DATA3='cnsm_exp2_2'

ENCODER1='none'
ENCODER2='rnn'
ENCODER3='bidirectionalrnn'
ENCODER4='transformer'

./ad_run.sh 5 $ENCODER1 $DATA1 32 -1 0.0256 SGD mean -1 -1 1>/dev/null 2>/dev/null & 
./ad_run.sh 5 $ENCODER1 $DATA2 32 -1 0.0732 SGD mean -1 -1 1>/dev/null 2>/dev/null &
./ad_run.sh 5 $ENCODER1 $DATA3 64 -1 0.0992 SGD mean -1 -1 1>/dev/null 2>/dev/null &

./ad_run.sh 4 $ENCODER2 $DATA1 128 256 0.0003 RMSprop max -1 -1 1>/dev/null 2>/dev/null &
./ad_run.sh 4 $ENCODER2 $DATA2 64 128 0.0319 SGD max -1 -1 1>/dev/null 2>/dev/null &
./ad_run.sh 4 $ENCODER2 $DATA3  32 32 0.0816 SGD mean -1 -1 1>/dev/null 2>/dev/null &

./ad_run.sh 3 $ENCODER3 $DATA1 128 256 0.0002 RMSprop max -1 -1 1>/dev/null 2>/dev/null &
./ad_run.sh 3 $ENCODER3 $DATA2  64 128 0.505 SGD last_hidden -1 -1 1>/dev/null 2>/dev/null &
./ad_run.sh 3 $ENCODER3 $DATA3 64 256 0.0448 SGD max -1 -1 1>/dev/null 2>/dev/null &

./ad_run.sh 2 $ENCODER4 $DATA1 32 -1 0.0157 SGD max 11 128 1>/dev/null 2>/dev/null &
./ad_run.sh 2 $ENCODER4 $DATA2 128 -1 0.0157 SGD max 1 128 1>/dev/null 2>/dev/null &
./ad_run.sh 2 $ENCODER4 $DATA3 64 -1 0.003 Adam max 1 64 1>/dev/null 2>/dev/null &
