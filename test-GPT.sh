#!/bin/bash

domain=$1
rank=$2
epochs=$3
device=$4
fold=${5:-0}
interval=${6:-100}

if [ "$domain" = "NavierStockPRec" ] || [ "$domain" = "NavierStockURec" ]; then
    python run-GPT-3D.py -domain=$domain \
                         -fold=$fold -h_dim=$rank -max_epochs=$epochs -device=$device \
                         -test_interval=$interval
else
    python run-GPT-2D.py -domain=$domain \
                         -fold=$fold -h_dim=$rank -max_epochs=$epochs -device=$device \
                         -test_interval=$interval                                          
fi

