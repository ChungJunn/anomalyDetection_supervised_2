#!/bin/bash

DATA1='cnsm_exp1'
DATA2='cnsm_exp2_1'
DATA3='cnsm_exp2_2'

REDUCE1='max'
REDUCE2='mean'

OPTIMIZER='Adam'
LR=0.001

./ad_run.sh 0 $DATA1 $REDUCE1 $OPTIMIZER $LR 1>/dev/null 2>/dev/null &
./ad_run.sh 0 $DATA2 $REDUCE1 $OPTIMIZER $LR 1>/dev/null 2>/dev/null &
./ad_run.sh 0 $DATA3 $REDUCE1 $OPTIMIZER $LR 1>/dev/null 2>/dev/null &

./ad_run.sh 1 $DATA1 $REDUCE2 $OPTIMIZER $LR 1>/dev/null 2>/dev/null &
./ad_run.sh 1 $DATA2 $REDUCE2 $OPTIMIZER $LR 1>/dev/null 2>/dev/null &
./ad_run.sh 1 $DATA3 $REDUCE2 $OPTIMIZER $LR 1>/dev/null 2>/dev/null &
