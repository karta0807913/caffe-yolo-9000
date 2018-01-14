#!/usr/bin/env sh

CAFFE_HOME=../..

SOLVER=voc_gnet_solver.prototxt
LOGFILE=logs/voc_log.log
#SNAPSHOT=--snapshot=voc_model/voc_model_iter_2410000.solverstate
SNAPSHOT=

$CAFFE_HOME/build/tools/caffe train $SNAPSHOT \
    --solver=$SOLVER --gpu=0 2>&1 | tee $LOGFILE
