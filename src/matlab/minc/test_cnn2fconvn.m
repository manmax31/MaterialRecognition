% Test conversion of a CNN to a fully convolutional network. 
clear;

addpath(genpath('/usr/local/caffe/caffe-master/matlab'));
caffe.set_mode_gpu();
caffe.set_device(1); % use K40 for large memory

cnn_model_file = {'/srv/datasets/Materials/OpenSurfaces/minc-model/deploy-alexnet.prototxt', ...
                  '/srv/datasets/Materials/OpenSurfaces/minc-model/deploy-googlenet.prototxt', ...
                  '/srv/datasets/Materials/OpenSurfaces/minc-model/deploy-vgg16.prototxt', ...
                 };

cnn_weight_file = {'/srv/datasets/Materials/OpenSurfaces/minc-model/minc-alexnet.caffemodel', ...
                   '/srv/datasets/Materials/OpenSurfaces/minc-model/minc-googlenet.caffemodel', ...
                   '/srv/datasets/Materials/OpenSurfaces/minc-model/minc-vgg16.caffemodel', ...
                  };

fconvn_model_file  = {'minc_alexnet_full_conv.prototxt', ...
                      'minc_googlenet_full_conv.prototxt', ...
                      'minc_vgg16_full_conv.prototxt'};
                  
fconvn_weight_file = {'../../results/minc/minc-alexnet-full-conv.caffemodel', ...
                      '../../results/minc/minc_googlenet_full_conv.caffemodel', ...
                      '../../results/minc/minc_vgg16_full_conv.caffemodel'};


% the original FC layers to be converted
fc_layers{1} = {'fc6', 'fc7', 'fc8-20'}; 
fc_layers{2} = {'fc8-20'};
fc_layers{3} = {'fc6', 'fc7', 'fc8-20'};

% the layers in the FCN corresponding to the fully connected layers
fconv_layers{1} = {'fc6-conv', 'fc7-conv', 'fc8-conv'}; 
fconv_layers{2} = {'fc8-conv'};
fconv_layers{3} = {'fc6-conv', 'fc7-conv', 'fc8-conv'};

% load data (for testing) and perform normalisation by channel
im_data(:, :, :, 1) = caffe.io.load_image('/srv/datasets/Materials/OpenSurfaces/minc-2500/images/carpet/carpet_000001.jpg');
im_data(:, :, :, 2) = caffe.io.load_image('/srv/datasets/Materials/OpenSurfaces/minc-2500/images/carpet/carpet_000003.jpg');
im_data(:, :, :, 3) = caffe.io.load_image('/srv/datasets/Materials/OpenSurfaces/minc-2500/images/carpet/carpet_000004.jpg');
channel_mean = [104, 117, 124]; % mean subtracted per channel
nc = size(im_data, 3);
for c = 1:nc
    im_data(:, :, c, :) = im_data(:, :, c, :) - channel_mean(c);
end

% convert each network
for i = 2
    fcn = cnn2fconvn(cnn_model_file{i}, cnn_weight_file{i}, fc_layers{i}, fconvn_model_file{i}, fconvn_weight_file{i}, fconv_layers{i});
    
    % test: compute the predicted class label on a coarse map.
    fcn.blobs('data').reshape(size(im_data));
    res = fcn.forward({im_data});

    % get the features for each layer
    % conv1 = fcn.blobs('conv1').get_data();
    %conv2 = fcn.blobs('conv2').get_data();
    %conv3 = fcn.blobs('conv3').get_data();
    %conv4 = fcn.blobs('conv4').get_data();
    %conv5 = fcn.blobs('conv5').get_data();
    %fc6_conv = fcn.blobs('fc6-conv').get_data();
    %fc7_conv = fcn.blobs('fc7-conv').get_data();
    fc8_conv = fcn.blobs('fc8-conv').get_data();
    prob = fcn.blobs('prob').get_data();

    % swap order between the row and column dimensions
    prob = permute(prob, [2 1 3 4]);
    [map_h, map_w, nc, n_imgs] = size(prob);

    % obtain the class with the maximum prob
    [max_prob, max_class] = max(prob, [], 3);
    max_prob = reshape(max_prob, [map_h, map_w, n_imgs]);
    max_class = reshape(max_class, [map_h, map_w, n_imgs]);
end

caffe.reset_all();

