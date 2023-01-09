#!/bin/bash

#export NCCL_IB_DISABLE=1
#export NCCL_SOCKET_IFNAME=eth0

source env.sh

CONFIG=$1
STAGE=data
MODE=train
WORK=home

export CODE_DIR=$DATA_HOME/code/dvq_charts

LOG_DIR="$PWD/log"
mkdir -p $LOG_DIR

LOG_FILE="$LOG_DIR/$1.log"
exec 1>> $LOG_FILE 2>&1

# The first hostname is the master address
export MASTER_ADDR=localhost
export MASTER_PORT=29501

cd $CODE_DIR

deepspeed --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --no_ssh_check \
  main.py -c $CONFIG -w $WORK -s $STAGE  -m $MODE

exit 0