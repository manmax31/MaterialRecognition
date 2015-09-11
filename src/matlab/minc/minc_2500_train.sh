#!/usr/bin/env sh

CAFFE_DIR=/usr/local/caffe/caffe-master

$CAFFE_DIR/build/tools/caffe train \
    --solver=minc_2500_alexnet_solver_fc8_tuned.prototxt \
    --weights=/usr/local/caffe/caffe-master/models/bvlc_alexnet/bvlc_alexnet.caffemodel \
    -gpu 1 


