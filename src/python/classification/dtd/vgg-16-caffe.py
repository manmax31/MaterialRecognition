import numpy as np
import matplotlib.pyplot as plt
import caffe
import sys
from pprint import pprint

caffe.set_mode_gpu()
caffe.set_device(0)

# MODEL_FILE = '/home/manab/Desktop/VGG/VGG_19_DTD_deploy.prototxt'
# PRETRAINED = '/srv/datasets/Materials/DTD/dtd-r1.0.1/dtd/output/models/scaled_384/4/scaled_384_4__iter_3600.caffemodel'
# CATEGORIES = [line.strip() for line in open('/srv/datasets/Materials/DTD/dtd-r1.0.1/dtd/output/Categories.txt', 'r')]

solver = caffe.SGDSolver('/home/manab/Desktop/GoogleNet_solver.prototxt')
pprint(solver.net.outputs)
