% Test the performance of the CNNs provided by Bell et al. (authors of MINC)
% by cropping the corner and center patches of each test image, scaled at
% different levels.
clear;

addpath(genpath('/home/huynh/Software/DeepLearning/Caffe/caffe-lmdb-matlab/matlab'));
% addpath(genpath('/home/huynh/Software/DeepLearning/Caffe/caffe-master/matlab'));
% addpath(genpath('/usr/local/caffe/caffe-master/matlab'));

caffe.set_mode_gpu();
caffe.set_device(1); % use K40 for large test batches (>= 100 images)

test_model = {'/srv/datasets/Materials/OpenSurfaces/minc-model/deploy-alexnet.prototxt', ...
              '/srv/datasets/Materials/OpenSurfaces/minc-model/deploy-googlenet.prototxt', ...
              '/srv/datasets/Materials/OpenSurfaces/minc-model/deploy-vgg16.prototxt', ...
             };

weights = {'/srv/datasets/Materials/OpenSurfaces/minc-model/minc-alexnet.caffemodel', ...
           '/srv/datasets/Materials/OpenSurfaces/minc-model/minc-googlenet.caffemodel', ...
           '/srv/datasets/Materials/OpenSurfaces/minc-model/minc-vgg16.caffemodel', ...
          };
      
net_name = {'alexnet', 'googlenet', 'vgg16'}; % names of different CNNs.

phase = 'test';

% LMDB database path
db_path = '/srv/datasets/Materials/OpenSurfaces/patch/lmdb/test_resized_db';
% db_path = '/srv/datasets/Materials/OpenSurfaces/minc-2500/minc-2500_test1_lmdb';

% result directory
res_dir = '../../results/minc';

for i = [1]
    fprintf('Testing %s\n', net_name{i});
    % net for loading test data
    test_net = caffe.Net(test_model{i}, weights{i}, phase); 

    % Perform testing    
    % [test_acc] = net_acc_crop_and_scale(test_net, db_path);
    [test_acc, test_conf_mat] = net_acc_crop_and_scale_v2(test_net, db_path);
    
    % [val_acc, val_loss] = net_acc(test_net, n_test_imgs);
    
    test_mean_class_acc = mean(diag(test_conf_mat));
    
    res_file = sprintf('%s/%s_Bell_et_al_crop_scale_acc.mat', res_dir, net_name{i});    
    
    % save(res_file, 'test_acc', 'test_loss', '-append');
    save(res_file, 'test_acc', 'test_conf_mat', 'test_mean_class_acc', '-append');
    
    % save(res_file, 'val_acc', 'val_loss', '-append');
    
    clear test_net;
end

caffe.reset_all();


