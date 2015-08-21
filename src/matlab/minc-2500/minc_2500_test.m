clear;

addpath(genpath('/usr/local/caffe/caffe-master/matlab'));

% trained model
model = '../../data/minc-2500/models/minc_2500_alexnet_train_val.prototxt';
weights = '../../data/minc-2500/models/minc_2500_alexnet_train1_iter_100000.caffemodel';

caffe.set_mode_gpu();
caffe.set_device(1);

% net = caffe.Net(model, weights, 'train'); % create net for training
test_net = caffe.Net(model, weights, 'test'); % create net for testing purposes

% Extract a test net referred to by a solver file.
solver = caffe.Solver('../../data/minc-2500/models/minc_2500_solver.prototxt');
% test_net = solver.test_nets(1);

% Load all the parameters from the trained CNN.
% test_net.copy_from(weights);

% Run test on the test samples preloaded in the model file.
test_net.forward_prefilled();

% Check input
data = test_net.blobs('data').get_data();
label = test_net.blobs('label').get_data();

% To recover the first input image of the last test batch
img1 = data(:, :, :, 1); % 4D blobs coming from Caffe have dimensions 
% in the order of [width, height, channels, num], where width is 
% the fastest dimension.

% swap width and height dimension 
img_t = permute(img1, [2 1 3]); 

% add the mean per channel (specified during training) back 
img_t(:, :, 1) = img_t(:, :, 1) + 104;
img_t(:, :, 2) = img_t(:, :, 2) + 117;
img_t(:, :, 3) = img_t(:, :, 3) + 124;
img_t = img_t(:, :, [3 2 1]); % swap from [B G R] channel order to [R G B] for display
% imtool(uint8(img_t));

% Obtain intermediate layers' parameters
conv1 = test_net.blobs('conv1').get_data();
conv1_layer = test_net.layers('conv1');
conv1_weights = conv1_layer.params(1, 1).get_data();
conv1_bias = conv1_layer.params(1, 2).get_data();


% Check output
result = cell(length(test_net.outputs), 1);
for n = 1:length(test_net.outputs)
    result{n} = test_net.blobs(test_net.outputs{n}).get_data();
end

