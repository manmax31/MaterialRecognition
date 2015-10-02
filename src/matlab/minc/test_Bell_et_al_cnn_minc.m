% Test the performance of the CNNs provided by Bell et al. (authors of MINC)
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

% Load test net model
test_model = {'minc_alexnet_test_Bell_et_al.prototxt', ...
              'minc_googlenet_test.prototxt', ...
              'minc_vgg16_test.prototxt', ...
             };

weights = {'/srv/datasets/Materials/OpenSurfaces/minc-model/minc-alexnet.caffemodel', ...
           '/srv/datasets/Materials/OpenSurfaces/minc-model/minc-googlenet.caffemodel', ...
           '/srv/datasets/Materials/OpenSurfaces/minc-model/minc-vgg16.caffemodel', ...
          };

net_name = {'alexnet', 'googlenet', 'vgg16'}; % names of different CNNs.
      
phase = 'test';

% result directory
res_dir = '../../results/minc';

% n_test_imgs = 5750; % minc_2500 test1
n_test_imgs = 299696; % minc test dataset
% n_test_imgs = 179795; % minc validation dataset

for i = [1]
    fprintf('Testing %s\n', net_name{i});
    % net for loading test data
    test_net = caffe.Net(test_model{i}, weights{i}, phase); 

    % Perform testing    
    % [test_acc, test_loss] = net_acc(test_net, n_test_imgs);
    % [val_acc, val_loss] = net_acc(test_net, n_test_imgs);
    
    [test_acc, test_loss, test_conf_mat] = net_acc_conf_mat(test_net, n_test_imgs);
    test_mean_class_acc = mean(diag(test_conf_mat));
    
    res_file = sprintf('%s/%s_Bell_et_al_acc.mat', res_dir, net_name{i});
    save(res_file, 'test_acc', 'test_loss', 'test_conf_mat', 'test_mean_class_acc', '-append');
    % save(res_file, 'val_acc', 'val_loss', '-append');
    
    % clear the nets
    clear test_net;
end

caffe.reset_all();    