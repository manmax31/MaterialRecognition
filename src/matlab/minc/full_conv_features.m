%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Compute the feature maps at every conv layer of a fully convolutional 
% network when a given image is passed through this network. 
% The network may have been converted from a CNN.
% 
% This functions computes the activation map for crops of the given image
% with the top-left corner located at multiples of a shift interval, then 
% stitches/interlaces the feature maps of these cropped images
% into a spatial map.
% 
% The input image size (height and width) does not need to match the input
% size of data layer of the fully conv net.
% 
% Input:
%   fcn: the Caffe net object representing a CNN.
% 
%   output_stride: stride of the output (class score, or prob) layer
%       i.e. 32 for AlexNet.
% 
%   shift_inter: shift interval in the image domain.
% 
%   I: the input (colour or gray) image (dimension height x width x nc).
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
%       layer names and the values are arrays storing the (activation) 
%       feature maps at the corresponding layers.
%   
%   Each feature maps is a 3D arrays with a size of 
%       f_map_height x f_map_width x n_filters, 
%       where f_map_height and f_map_width are the height and width of the 
%       feature map in each layer and n_filter is the number of 
%       convolution filters or output neurons per local receptive field.
%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [m] = full_conv_features(fcn, output_stride, shift_inter, img, blobs, channel_mean)
    [h, w, nc] = size(img);
    
    % number of image samples in each dimension (x and y)
    n_samples = floor(output_stride/shift_inter);
    
    % put all the shifted version into an image stack
    imgs = zeros(h, w, nc, n_samples^2);
    shift_idx = 0; % index of each shifted image
    for i = 1:shift_inter:output_stride 
        for j = 1:shift_inter:output_stride 
            shift_idx = shift_idx + 1;
            shifted_I = img(i:end, j:end, :);
            % pad the shifted image to arrive at the same dimension as the
            % original image.
            imgs(:, :, :, shift_idx) = padarray(shifted_I, [i-1 j-1], 0, 'post');
        end
    end

    % convert image data to a Caffe compatible format
    imgs = mat_2_caffe_img(imgs, channel_mean);

    % Forward pass to create convnet features in every layer
    fcn.blobs('data').reshape(size(imgs));
    res = fcn.forward({imgs});
    
    % a map storing entries with keys corresponding to blob_names
    % and values corresponding to activation features for that layer.
    m = containers.Map; 
    
    % extract features for each blob 
    n_blobs = numel(blobs);
    for i_blob = 1:n_blobs
        blob_name = blobs{i_blob};
        
        % the activation features in 4D
        f_4d = fcn.blobs(blob_name).get_data();
        
        % swap order between the row and column dimensions
        f_4d = permute(f_4d, [2 1 3 4]);
        
        % interlace the features map spatially (stitching) into a 3D cube
        [map_h, map_w, nc, n_imgs] = size(f_4d);
        f_3d = zeros(map_h * n_samples, map_w * n_samples, nc);
        shift_idx = 0;
        for i = 1:n_samples
            for j = 1:n_samples
                shift_idx = shift_idx + 1;
                f_3d([0:n_samples:(map_h-1)*n_samples]+i, [0:n_samples:(map_w-1)*n_samples]+j, :) ...
                = f_4d(:, :, :, shift_idx);
            end
        end
        m(blob_name) = f_3d;
    end
end
