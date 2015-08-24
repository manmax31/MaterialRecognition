__author__ = 'manabchetia'

import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
# caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
# sys.path.insert(0, caffe_root + 'python')
sys.path.append("/Users/manabchetia/Downloads/caffe/python/")
import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '../data/MINC/minc-model/deploy-googlenet.prototxt'
PRETRAINED = '../data/MINC/minc-model/minc-googlenet.caffemodel'
IMAGE_FILE = '../data/MINC/minc-2500/images/mirror/mirror_000008.jpg'

caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       channel_swap=(2, 1, 0),
                       raw_scale=255,
                       image_dims=(362, 362))
input_image = caffe.io.load_image(IMAGE_FILE)
plt.imshow(input_image)

prediction = net.predict(
    [input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
print 'prediction shape:', prediction[0].shape
plt.plot(prediction[0])
print 'predicted class:', prediction[0].argmax()
# plt.show()
