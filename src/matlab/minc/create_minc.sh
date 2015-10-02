#!/usr/bin/env sh
# Create the minc-2500 lmdb inputs
# N.B. set the path to the minc-2500 train + val data dirs

TOOLS=/usr/local/caffe/caffe-master/build/tools
DATA_ROOT=/
LABEL_DIR=/srv/datasets/Materials/OpenSurfaces/patch/lmdb

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
  echo "Set the TRAIN_DATA_ROOT variable in create_minc.sh to the path" \
       "where the MINC images are stored."
  exit 1
fi

#echo "Creating train lmdb..."

#GLOG_logtostderr=1 $TOOLS/convert_imageset \
##    --resize_height=$RESIZE_HEIGHT \
##    --resize_width=$RESIZE_WIDTH \
#    --shuffle \
#    $DATA_ROOT \
#    $LABEL_DIR/train_balanced.txt \
#    $LABEL_DIR/train_balanced_db

#echo "Creating val lmdb..."

#GLOG_logtostderr=1 $TOOLS/convert_imageset \
##    --resize_height=$RESIZE_HEIGHT \
##    --resize_width=$RESIZE_WIDTH \
#    --shuffle \
#    $DATA_ROOT \
#    $LABEL_DIR/val.txt \
#    $LABEL_DIR/val_db

echo "Creating test lmdb..."

echo "Data root $DATA_ROOT"
echo "Label dir $LABEL_DIR"

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    --encode_type jpg \
    $DATA_ROOT \
    $LABEL_DIR/test.txt \
    $LABEL_DIR/test_db_resized

chmod -R 777 $LABEL_DIR/test_db_resized

echo "Done."

