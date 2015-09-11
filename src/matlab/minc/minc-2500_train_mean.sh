#!/usr/bin/env sh
# Compute the mean image from the minc-2500 training lmdb

TOOLS=/usr/local/caffe/caffe-master/build/tools

# Directory containing both the LMDB training file 
# which is also the destination of the mean image file.
DATA=../../data/minc/minc-2500

$TOOLS/compute_image_mean $DATA/minc-2500_train1_lmdb \
  $DATA/minc-2500_train1_mean.binaryproto

echo "Done."
