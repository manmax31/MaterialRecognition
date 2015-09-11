% Test the generation of CNN features from a test image.
clear;
addpath(genpath('/usr/local/caffe/caffe-master/matlab'));

caffe.set_mode_gpu();
caffe.set_device(0);

model = 'minc_2500_alexnet_test.prototxt';
iter = 4000;
weights = sprintf('../../results/minc_2500/alexnet/27_08_2015/minc_2500_alexnet_train1_27Aug2015_v1_iter_%d.caffemodel', iter);

net = caffe.Net(model, 'test'); % create net for testing purposes
net.copy_from(weights);

img = imread('/srv/datasets/Materials/OpenSurfaces/minc-2500/images/brick/brick_001667.jpg');

[f] = cnn_features(net, img);

clear net;