% Test the performance of the converted FCNs from the CNNs provided by 
% Bell et al. (authors of MINC)
% by sliding the input window of these FCNs over the test images.
clear;

addpath(genpath('/home/huynh/Software/DeepLearning/Caffe/caffe-lmdb-matlab/matlab'));
% addpath(genpath('/home/huynh/Software/DeepLearning/Caffe/caffe-master/matlab'));
% addpath(genpath('/usr/local/caffe/caffe-master/matlab'));

caffe.set_mode_gpu();
caffe.set_device(1); % use K40 for large test batches (>= 100 images)
      
test_model  = {'minc_alexnet_full_conv.prototxt', ...
                      'minc_googlenet_full_conv.prototxt', ...
                      'minc_vgg16_full_conv.prototxt'};
                  
weights = {'../../results/minc/minc-alexnet-full-conv.caffemodel', ...
                      '../../results/minc/minc_googlenet_full_conv.caffemodel', ...
                      '../../results/minc/minc_vgg16_full_conv.caffemodel'};      
      
net_name = {'alexnet', 'googlenet', 'vgg16'}; % names of different CNNs.

phase = 'test';

% LMDB database path
db_path = '/srv/datasets/Materials/OpenSurfaces/patch/lmdb/test_db_resized';
% db_path = '/srv/datasets/Materials/OpenSurfaces/minc-2500/minc-2500_test1_lmdb';

batch_size = [128; 128; 64];

% result directory
res_dir = '../../results/minc';

for i = [1]
    fprintf('Testing %s\n', net_name{i});
    
    % net for loading test data
    test_net = caffe.Net(test_model{i}, weights{i}, phase); 

    % Perform testing    
    [test_acc, test_conf_mat] = net_acc_fcn(test_net, db_path, batch_size(i));
       
    test_mean_class_acc = mean(diag(test_conf_mat));
    
    res_file = sprintf('%s/%s_Bell_et_al_fcn_resized_jpg_acc.mat', res_dir, net_name{i});
    
    save(res_file, 'test_acc', 'test_conf_mat', 'test_mean_class_acc');
    
    clear test_net;
end

caffe.reset_all();


