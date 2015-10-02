% Test the performance of our trained CNN models on the MINC dataset.
% 
% BIG NOTE:
% 
% In order to use Net.forward_prefilled() in the Caffe Matlab interface, 
% the layer names between the train/val and test network prototxt files
% need to match exactly. 
% 
% Changing the names of the input (bottom) and output (top) blobs of each
% layer is fine, as long as the network architectures used for training
% and testing are the same. 
% 
clear;
addpath(genpath('/usr/local/caffe/caffe-master/matlab'));

caffe.set_mode_gpu();
caffe.set_device(1); % use K40 for large test batches (>= 100 images)

% Load CNN models of various kinds (Alexnet, Googlenet and VGG16)
test_model = {'minc_alexnet_test_ours.prototxt', ...
              %'minc_googlenet_test.prototxt', ...
              %'minc_vgg16_test.prototxt', ...
             };

% our trained models.
iter = 30000;
weights = {sprintf('../../results/minc_2500/alexnet/27_08_2015/minc_2500_alexnet_train1_27Aug2015_v1_iter_%d.caffemodel', iter), ...
            %'/srv/datasets/Materials/OpenSurfaces/minc-model/minc-googlenet.caffemodel', ...
            %'/srv/datasets/Materials/OpenSurfaces/minc-model/minc-vgg16.caffemodel', ...
          };

net_name = {'alexnet', 'googlenet', 'vgg16'}; % names of different CNNs.
      
phase = 'test';

% result directory
res_dir = '../../results/minc';

% n_test_imgs = 5750; % minc_2500 test1
n_test_imgs = 299696; % minc test dataset
% n_test_imgs = 179795; % minc validation dataset

for i = 1
    fprintf('Testing %s\n', net_name{i});
    
    % test net with the test dataset preloaded
    test_net = caffe.Net(test_model{i}, weights{i}, phase); 

    % Perform testing
    % [test_acc, test_loss] = net_acc(test_net, n_test_imgs);
    [test_acc, test_loss, test_conf_mat] = net_acc_conf_mat(test_net, n_test_imgs);
    test_mean_class_acc = mean(diag(test_conf_mat));
    
    res_file = sprintf('%s/%s_minc_2500_train1_27Aug2015_v1_iter_%d_acc.mat', res_dir, net_name{i}, iter);
    save(res_file, 'test_acc', 'test_loss', 'test_conf_mat', 'test_mean_class_acc', '-append');
    % save(res_file, 'val_acc', 'val_loss', '-append');
    
    % clear the nets
    clear test_net;
end
caffe.reset_all();


% img = imread('/srv/datasets/Materials/OpenSurfaces/minc-2500/images/brick/brick_001667.jpg');
% blobs = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7'};
% channel_mean = [104, 117, 124];
% [m] = cnn_features(test_net, img, blobs, channel_mean);
% clear test_net;



