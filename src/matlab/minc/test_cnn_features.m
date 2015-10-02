% Test the generation of CNN features from a test image.
clear;
addpath(genpath('/usr/local/caffe/caffe-master/matlab'));

caffe.set_mode_gpu();
caffe.set_device(0);

model = 'minc_2500_alexnet_deploy.prototxt';
iter = 4000;
weights = sprintf('../../results/minc_2500/alexnet/27_08_2015/minc_2500_alexnet_train1_27Aug2015_v1_iter_%d.caffemodel', iter);

phase = 'test';
net = caffe.Net(model, weights, phase); % create net for testing purposes

img = imread('/srv/datasets/Materials/OpenSurfaces/minc-2500/images/brick/brick_001667.jpg');

blobs = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7'};

channel_mean = [104, 117, 124];

[m] = cnn_features(net, img, blobs, channel_mean);

clear net;

caffe.reset_all();