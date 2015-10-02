clear;

addpath(genpath('/usr/local/caffe/caffe-master/matlab'));

caffe.set_mode_gpu();
caffe.set_device(0);

VGG_model_dir = '/usr/local/caffe/caffe-master/models/vgg_ilsvrc14/';

test_model = {[VGG_model_dir, 'VGG_ILSVRC_16_layers_deploy.prototxt'], ...
              [VGG_model_dir, 'VGG_ILSVRC_19_layers_deploy.prototxt']};

weights = {[VGG_model_dir, 'VGG_ILSVRC_16_layers.caffemodel'], ...
           [VGG_model_dir, 'VGG_ILSVRC_19_layers.caffemodel']};

phase = 'test';

for i = [1 2]
    test_net = caffe.Net(test_model{i}, weights{i}, phase);
end

caffe.reset_all();