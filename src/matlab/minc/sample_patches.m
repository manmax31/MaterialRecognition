%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sample rectangular patches from an image at regular horizontal and 
% vertical strides.
% 
% Input: 
%   img: the input colour image, stored as an array of size height x width
%       x n_channels.
%   patch_size: a two element vector containing the patch height and width.
%   stride: a two element vector containing the vertical and horizontal 
%       strides, respectively, between patches.
% 
% Output: 
%   patches: image patches, stored as an array of size height x width x
%       n_channels x n_patches.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
function patches = sample_patches(img, patch_size, stride)
    [img_h, img_w, n_channels] = size(img);
    patch_h = patch_size(1);
    patch_w = patch_size(2);
    stride_y = stride(1);
    stride_x = stride(2); 
    
    % central locations of the patches
    [centre_x, centre_y] = meshgrid(...
        [floor(patch_w/2)+1:stride_x:img_w-floor(patch_w/2)-1, img_w-floor(patch_w/2)], ...
        [floor(patch_h/2)+1:stride_y:img_h-floor(patch_h/2)-1, img_h-floor(patch_h/2)]);
    
    % sample patches from the input image 
    n_patches = numel(centre_x);
    patches = zeros(patch_h, patch_w, n_channels, n_patches);
    centre_x = centre_x(:);
    centre_y = centre_y(:);
    for patch_count = 1:n_patches
        y = centre_y(patch_count);
        x = centre_x(patch_count);
        patches(:, :, :, patch_count) = ...
            img(y-floor(patch_h/2):y-floor(patch_h/2)+patch_h-1, ...
                x-floor(patch_w/2):x-floor(patch_w/2)+patch_w-1, :);
    end
end