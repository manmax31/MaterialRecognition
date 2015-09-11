#!/usr/bin/env sh

CAFFE_DIR=/usr/local/caffe/caffe-master

$CAFFE_DIR/build/tools/caffe train \
    --solver=../../data/minc-2500/models/minc_2500_solver.prototxt\
    --snapshot=../../data/minc-2500/models/minc_2500_alexnet_train1_iter_440000.solverstate
    -gpu 0
