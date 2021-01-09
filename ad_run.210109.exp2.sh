#!/bin/bash

DATA1='cnsm_exp1'
DATA2='cnsm_exp2_1'
DATA3='cnsm_exp2_2'

REDUCE1='max'
REDUCE2='mean'
REDUCE3='last_hidden'

./ad_run.sh 0 $DATA1 $REDUCE1 1>/dev/null 2>/dev/null &
./ad_run.sh 0 $DATA2 $REDUCE1 1>/dev/null 2>/dev/null &
./ad_run.sh 0 $DATA3 $REDUCE1 1>/dev/null 2>/dev/null &

./ad_run.sh 1 $DATA1 $REDUCE2 1>/dev/null 2>/dev/null &
./ad_run.sh 1 $DATA2 $REDUCE2 1>/dev/null 2>/dev/null &
./ad_run.sh 1 $DATA3 $REDUCE2 1>/dev/null 2>/dev/null &

./ad_run.sh 2 $DATA1 $REDUCE3 1>/dev/null 2>/dev/null &
./ad_run.sh 2 $DATA2 $REDUCE3 1>/dev/null 2>/dev/null &
./ad_run.sh 2 $DATA3 $REDUCE3 1>/dev/null 2>/dev/null &
