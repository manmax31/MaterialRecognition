# Ref: https://groups.google.com/forum/#!msg/caffe-users/6OOcM-XfvOI/Cs5VVdfDubEJ
# Ref: https://github.com/BVLC/caffe/issues/745
import lmdb
import numpy as np
import matplotlib.image as mpimg
from collections import defaultdict
from skimage.transform import resize
import sys

import caffe
caffe.set_mode_gpu()

MODEL_FILE = '/home/manab/Desktop/VGG/VGG_19_DTD_deploy.prototxt'
PRETRAINED = '/home/manab/Downloads/VGG_Caffe_Model/VGG_ILSVRC_16_layers.caffemodel'
db_path = '/home/manab/Documents/lmdb/train_db/'

count = 0
correct = 0
matrix = defaultdict(int) # (real,pred) -> int
labels_set = set()

net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

lmdb_env = lmdb.open(db_path)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()


count = 0

for key, value in lmdb_cursor:
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(value)
    label = int(datum.label)
    image = caffe.io.datum_to_array(datum)
    # print 'Orig:', image.shape

    image = image[:, :, (2, 1, 0)]
    image = image.transpose((1, 2, 0))
    image = image.astype(np.uint8)
    # print 'Transpose:', image.shape

    image = resize(image, (224, 224))
    image = image.transpose((2, 1, 0))
    # print 'Resized:', image.shape

    # count += 1
    #
    # if count == 1:
    #     break


    out = net.forward_all(data=np.asarray([image]))
    plabel = int(out['prob'][0].argmax(axis=0))

    count = count + 1
    iscorrect = label == plabel
    correct = correct + (1 if iscorrect else 0)
    matrix[(label, plabel)] += 1
    labels_set.update([label, plabel])

    if not iscorrect:
        print("\rError: key=%s, expected %i but predicted %i" \
                % (key, label, plabel))

    sys.stdout.write("\rAccuracy: %.1f%%" % (100.*correct/count))
    sys.stdout.flush()

print(str(correct) + " out of " + str(count) + " were classified correctly")

print ""
print "Confusion matrix:"
print "(r , p) | count"
for l in labels_set:
    for pl in labels_set:
        print "(%i , %i) | %i" % (l, pl, matrix[(l,pl)])






# lmdb_cursor.get('{:0>10d}'.format(1))  # get the data associated with the 'key' 1, change the value to get other images
# value = lmdb_cursor.value()
# key = lmdb_cursor.key()
#
# print(value)
#
# datum = caffe.proto.caffe_pb2.Datum()
# datum.ParseFromString(value)
# image = np.zeros((datum.channels, datum.height, datum.width))
# image = caffe.io.datum_to_array(datum)
# image = np.transpose(image, (1, 2, 0))
# image = image[:, :, (2, 1, 0)]
# image = image.astype(np.uint8)
#
# mpimg.imsave('out.png', image)
