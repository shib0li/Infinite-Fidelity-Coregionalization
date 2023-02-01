#!/bin/bash

domain=$1
rank=$2
epochs=$3
device=$4
fold=${5:-0}
interval=${6:-100}
depthA=${7:-2}

if [ "$domain" = "NavierStockPRec" ] || [ "$domain" = "NavierStockURec" ] || [ "$domain" = "NavierStockVRec" ]; then
    python run-ODE-3D.py -domain=$domain \
                         -fold=$fold -h_dim=$rank -max_epochs=$epochs -device=$device \
                         -A_depth=$depthA -test_interval=$interval                     
else
    python run-ODE-2D.py -domain=$domain \
                         -fold=$fold -h_dim=$rank -max_epochs=$epochs -device=$device \
                         -A_depth=$depthA -test_interval=$interval
fi

                     