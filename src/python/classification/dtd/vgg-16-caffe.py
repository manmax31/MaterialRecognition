import lmdb
import numpy as np
import matplotlib.image as mpimg


import caffe

db_path = '../data/liris-accede/train_data_lmdb/'

lmdb_env = lmdb.open(db_path)  # equivalent to mdb_env_open()
lmdb_txn = lmdb_env.begin()  # equivalent to mdb_txn_begin()
lmdb_cursor = lmdb_txn.cursor()  # equivalent to mdb_cursor_open()
lmdb_cursor.get('{:0>10d}'.format(1))  # get the data associated with the 'key' 1, change the value to get other images
value = lmdb_cursor.value()
key = lmdb_cursor.key()

datum = caffe.proto.caffe_pb2.Datum()
datum.ParseFromString(value)
image = np.zeros((datum.channels, datum.height, datum.width))
image = caffe.io.datum_to_array(datum)
image = np.transpose(image, (1, 2, 0))
image = image[:, :, (2, 1, 0)]
image = image.astype(np.uint8)

mpimg.imsave('out.png', image)
