%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Extract CNN features of an image as output by the given layers.
% 
% These layers are typically convolutional (conv) and 
%   fully connected (fc) layers.
% 
% The input image size (height and width) does not need to match the input
% size of data layer of the CNN.
% 
% Input:
%   net: the Caffe net object representing a CNN.
% 
%   img: the input (colour or gray) image.
% 
%   blobs: a cell array storing the names of the output blobs whose 
%       (output) data are extracted as features of the given image.
% 
%   channel_mean: a 3-element array storing the means of 
%       the B, G, R channels to be subtracted from the input images 
%       (in that order).
% 
% Output:  
%   m: a Map object containing (key, value) entry pairs, where the keys are 
%       layer names and the values are arrays storing the feature maps 
%       as a result of network forwarding through the corresponding layers.
%   
%   The structure of the feature maps for each layer is as follows
%   
%   - convolutional layers: the feature maps are 4D arrays with a size 
%       of f_map_height x f_map_width x n_filters x n_patches, 
%       where f_map_height and f_map_width are the number of rows and 
%       columns of neurons in the convolutional layer, n_features is 
%       the length of the feature vector (also number of convolutional 
%       filters or output neurons), and n_patches is the number of 
%       patches sampled from the image.
% 
%   - fully connected layers: the feature maps are 2D arrays with a size of 
%       n_features x n_patches, where n_features is the number of output
%       neurons (features), and n_patches is the number of patches 
%       sampled from the image.
%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [m] = cnn_features(net, img, blobs, channel_mean)
    [img_h, img_w, n_channels] = size(img);

    % input size required by the net
    input_size = net.blobs('data').shape;
    patch_h = input_size(1); % patch height
    patch_w = input_size(2); % patch width
    assert(n_channels == input_size(3));
    
    % set the number of patches in a single batch.
    batch_size = 100;
    
    % the horizontal and vertical stride between successively sampled patches
    conv1_weights = net.layers('conv1').params(1).get_data();
    [conv1_w, conv1_h, ~, ~] = size(conv1_weights); % dimension of the conv1 filter.
    
    % the x and y strides of the conv1 filters in the image domain
    % conv1_stride_x = 4;
    % conv1_stride_y = 4;
    
    % the stride allows an overlapping of (conv1_w - conv1_stride_x) pixels 
    % along the width and (conv1_h - conv1_stride_y) pixels along the
    % height.
    % stride_x = patch_w - (conv1_w - conv1_stride_x);
    % stride_y = patch_h - (conv1_h - conv1_stride_y);
    stride_x = 16;
    stride_y = 16;    
    
    % sample patches that matches the CNN input size.
    patches = sample_patches(img, [patch_h patch_w], [stride_y stride_x]);
    
    % preprocess the patches to conform to Caffe format
    patches = mat_2_caffe_img(patches, channel_mean);
    
    % Use this instead of mat_2_caffe_img for Bell et al.'s nets trained 
    % on MINC (done via the Python interface).
    % target_height = 227;
    % target_width = 227;    
    % patches = py_2_caffe_img(patches, target_height, target_width, ...
    %    channel_mean, width_flip);
    
    n_blobs = numel(blobs);    
    m = containers.Map; 
    
    % go through each batch of patches
    n_patches = size(patches, 4);
    n_batches = ceil(n_patches/batch_size); % number of patch batches 
    for i = 1:n_batches
        batch = patches(:, :, :, (i-1)*batch_size+1 : min(i*batch_size, n_patches));  
        
        % update the number of patches in the current batch
        n_batch_patches = size(batch, 4);         
        net.blobs('data').reshape([patch_h patch_w n_channels n_batch_patches]);
        
        % load the image into the net.
        res = net.forward({batch});

        % for each output blob
        for j = 1:n_blobs
            blob_name = blobs{j};
            blob = net.blobs(blob_name);
            blob_output = blob.get_data(); % in 4D size (width x height 
                                           % x n_filters x n_images)
                                           
            % add the output features to the map
            if (m.isKey(blob_name))
                % add to existing feature vectors for the current layer
                cur_blob_features = m(blob_name); % current feature vectors
                                                  % for the current blob
                if (ndims(cur_blob_features) == 4) % conv layer
                    new_blob_features = cat(4, cur_blob_features, blob_output);
                elseif (ismatrix(cur_blob_features)) % fc layer
                    new_blob_features = cat(2, cur_blob_features, blob_output);
                end
                m(blob_name) = new_blob_features;
            else
                % add the features for the new layer 
                m(blob_name) = blob_output;
            end
        end             
    end
    
end




