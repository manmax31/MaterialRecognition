#!/usr/bin/env sh
# Create the minc-2500 lmdb inputs
# N.B. set the path to the minc-2500 train + val data dirs

TOOLS=/usr/local/caffe/caffe-master/build/tools
DATA_ROOT=/srv/datasets/Materials/OpenSurfaces/minc-2500/

LABEL_DIR=../../data/minc/minc-2500

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "Error: DATA_ROOT is not a path to a directory: $DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA_ROOT \
    $LABEL_DIR/train1.txt \
    $LABEL_DIR/minc-2500_train1_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA_ROOT \
    $LABEL_DIR/validate1.txt \
    $LABEL_DIR/minc-2500_val1_lmdb

echo "Creating test lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA_ROOT \
    $LABEL_DIR/test1.txt \
    $LABEL_DIR/minc-2500_test1_lmdb

echo "Done."

