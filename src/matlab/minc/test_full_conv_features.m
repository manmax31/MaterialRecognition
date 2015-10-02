clear;
addpath(genpath('/usr/local/caffe/caffe-master/matlab'));

caffe.set_mode_gpu();
caffe.set_device(0);

fconvn_model_file  = 'minc_alexnet_full_conv.prototxt';
fconvn_weight_file = '../../results/minc/minc-alexnet-full-conv.caffemodel';

fcn = caffe.Net(fconvn_model_file, fconvn_weight_file, 'test');

% stride of the output (class score, or prob) layer
output_stride = 32;

% shift interval in the image domain
shift_inter = 8;

% img = imread('/srv/datasets/Materials/OpenSurfaces/minc-2500/images/brick/brick_001667.jpg');
img = imread('/srv/datasets/Materials/OpenSurfaces/minc-2500/images/carpet/carpet_000003.jpg');
% img = imread('/srv/datasets/Materials/OpenSurfaces/minc-2500/images/carpet/carpet_000004.jpg');
% img = imread('/srv/datasets/Materials/FMD/image/fabric/fabric_moderate_002_new.jpg');

blobs = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6-conv', 'fc7-conv', 'fc8-conv', 'prob'};

channel_mean = [104, 117, 124];

tic;
[m] = full_conv_features(fcn, output_stride, shift_inter, img, blobs, channel_mean);

% save the activation features 
% out_file = '../../results/minc/brick_001667_alexnet_features.mat';
out_file = '../../results/minc/carpet_000003_alexnet_features.mat';
% out_file = '../../results/minc/carpet_000004_alexnet_features.mat';
% out_file = '../../results/minc/fabric_moderate_002_new.mat';
save(out_file, 'm');
toc;

% obtain the class labels from the class probabilities
conv1 = m('conv1');
conv2 = m('conv2');
conv3 = m('conv3');
conv4 = m('conv4');
conv5 = m('conv5');
fc6_conv = m('fc6-conv');
fc7_conv = m('fc7-conv');
prob = m('prob');
[max_prob, class_label] = max(prob, [], 3);

clear net;
caffe.reset_all();

