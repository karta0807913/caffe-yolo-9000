#!/usr/bin/env sh

CAFFE_HOME=../..

SOLVER=gnet_solver.prototxt
LOGFILE=logs/voc_log.log

$CAFFE_HOME/build/tools/caffe train \
    --solver=$SOLVER --gpu=0 2>&1 | tee $LOGFILE
