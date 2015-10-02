%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Convert an image batch in Matlab with the size height x width x
% n_channels x n_images to the data in caffe's format: 
% width x height x 3 x n_images with BGR channels.
% 
% The function takes at least two arguments, the image batch and
% per-channel means, followed by optional arguments incl. the target size
% for resizing and a flag for creating horizontally mirrored copies.
% 
% Input: 
%   im_batch: image batches, stored as an array of size height x width x
%       n_channels x n_images.
%
%   channel_mean: a 3-element array storing the means of 
%       the B, G, R channels to be subtracted from the input images 
%       (in that order).
% 
%   The following inputs are optional: 
% 
%   target_height, target_width: the target height and width to resize to.
% 
%   width_flip: a boolean flip. If set, a copy of the original 
%       image batch is added, with array elements flipped along 
%       the image width.
% 
% Output:
%   out_data: the image batch in the Caffe format (with dimension width x 
%       height x n_channels x n_images if width_flip is false, 
%       or width x height x n_channels x (2*n_images) 
%       if width_flip is true).   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
function out_data = mat_2_caffe_img(im_batch, channel_mean, varargin)

    if (nargin < 2)
            error('mat_2_caffe_img accepts at least im_batch and channel_mean args.');
    end
    
    if (nargin >= 5)
        error('mat_2_caffe_img accepts 4 or less args.');
    end

    [height, width, n_c, n_patches] = size(im_batch);
    
    out_data = im_batch(:, :, [3, 2, 1], :);  % permute channels from RGB to BGR
    out_data = single(out_data);  % convert from uint8 to single

    % subtract mean_data (stored in 3 x 1 vector already in BGR order)
    out_data(:, :, 1, :) = out_data(:, :, 1, :) - channel_mean(1);
    out_data(:, :, 2, :) = out_data(:, :, 2, :) - channel_mean(2);
    out_data(:, :, 3, :) = out_data(:, :, 3, :) - channel_mean(3);
    
    % resize if requested
    if (nargin >= 3)
        target_size = varargin{1};
        target_height = target_size(1);
        target_width  = target_size(2); 
        % resize if the target size is different from the current size
        if (target_height ~= height || target_width ~= width)           
            out_data = reshape(out_data, [height, width, n_c * n_patches]); 
            out_data = imresize(out_data, [target_height, target_width], 'bilinear');  % resize im_data
            out_data = reshape(out_data, [target_height, target_width, n_c, n_patches]);
        end
    end

    % flip the image about the x-axis.
    if (nargin == 4 && varargin{2})
        out_data(:, :, :, n_patches+1:2*n_patches) = out_data(:, end:-1:1, :, 1:n_patches);
    end
    
    % flip width and height
    out_data = permute(out_data, [2, 1, 3, 4]);  
end
