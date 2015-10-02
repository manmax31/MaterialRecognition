% generate the activation feature maps of fully conv nets for FMD 
clear;
addpath(genpath('/usr/local/caffe/caffe-master/matlab'));

caffe.set_mode_gpu();
caffe.set_device(0);

% material categories
FMD_dir = '/srv/datasets/Materials/FMD/image';
d_list = dir(FMD_dir);
isub = [d_list(:).isdir]; 
mats = {d_list(isub).name}';
mats(ismember(mats,{'.','..'})) = [];

% directory contain material masks
FMD_mask_dir = '/srv/datasets/Materials/FMD/mask'; 

% output directory
out_dir = '../../results/FMD';
if (~exist(out_dir, 'dir'))
    if (exist(out_dir, 'file'))
        delete out_dir;
    end
    mkdir(out_dir);
end

% initiate the FCN
fconvn_model_file  = 'minc_alexnet_full_conv.prototxt';
fconvn_weight_file = '../../results/minc/minc-alexnet-full-conv.caffemodel';
fcn = caffe.Net(fconvn_model_file, fconvn_weight_file, 'test');

% stride of the output (class score, or prob) layer
output_stride = 32;

% shift interval in the image domain
shift_inter = 8;

% blobs at which to extract features
blobs = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6-conv', 'fc7-conv'};

% per-channel mean for subtraction
channel_mean = [104, 117, 124];

% generate activation features for each image
n_mats = numel(mats);
for m = 1:n_mats    

    if (m <= 2)
        continue;
    end
    
    fprintf('Material %s ...\n', mats{m});
    file_list = dir([FMD_dir, '/', mats{m}, '/*.jpg']);
    is_file = ~[file_list(:).isdir];
    file_names = {file_list(is_file).name}';    
    n_files = numel(file_names);
    
    % output dir for the material
    out_mat_dir = [out_dir, '/', mats{m}];
    if (~exist(out_mat_dir, 'dir'))
        if (exist(out_mat_dir, 'file'))
            delete out_mat_dir;
        end
        mkdir(out_mat_dir);
    end

    if (m == 3)
        n0 = 30;
    else
        n0 = 1;
    end
    for n = n0:n_files
        fprintf('File %s\n', file_names{n});
        file_name = [FMD_dir, '/', mats{m}, '/', file_names{n}];
        img = imread(file_name);
        [~, ~, nc] = size(img);
        
        % convert any grayscale to rgb
        if (nc == 1) 
           img = img(:, :, ones(3, 1));
        end
        
        % mark the selected materials
        mask_file_name = [FMD_mask_dir, '/', mats{m}, '/', file_names{n}]; 
        mask = imread(mask_file_name);        
        mask = uint8(mask(:, :, 1) > 0);
        for c = 1:nc
            img(:, :, c) = img(:, :, c) .* mask;
        end
        
        [features] = full_conv_features(fcn, output_stride, shift_inter, img, blobs, channel_mean);
        
        % output file, remove extension from the original name
        out_file = [out_dir, '/', mats{m}, '/', file_names{n}(1:end-4), '_alexnet.mat'];
        save(out_file, 'features');
    end
end

clear net;
caffe.reset_all();


