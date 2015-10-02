%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create an image stack by cropping patches of a given size 
% at the corners and centre of a given image.
% The patches are also sampled as different scales of the original image,
% given by a vector of scales.
% 
% Input:
%   img   : the given image, size h x w x nc.
%   p_size: patch height and width (in that order), size 2 x 1.
%   scales: scales of the input image other than 1 (the original scale), 
%       size n_scales x 1. Scales larger 1 means the input image is
%       enlarged before crop sampling, and smaller than 1 means the input
%       image is shrunk before cropping.
% 
% Output:
%   I_stack: the output image stack by concatenating image patches 
%       sampled at the above locations and scales, along the channel dimension.
%       The size of I_stack is h x w x nc x (n_scales * 5).
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [I_stack] = crop_and_scale(I, p_size, scales)
    % patch width and height
    ph = p_size(1);
    pw = p_size(2);
    nc = size(I, 3);
    
    if ~ismember(1, scales)
        scales = [1; scales];
    end
        
    % go through the given scales
    n_scales = size(scales, 1);
    I_stack = zeros(ph, pw, nc, n_scales * 5);
    
    for i = 1:n_scales        
        % resize image to the current scale
        J = imresize(I, scales(i)); 
        
        [h, w, ~] = size(J); % image height and width at the current scale
        
        errMsg = sprintf('Image size is smaller than patch size when scaled at %d.\n', scales(i));
        assert(ph <= h && pw <= w, errMsg);
        
        % center crop
        I_stack(:, :, :, 5*(i-1)+1) = J(floor((h-ph)/2) : floor((h-ph)/2)+ph-1, floor((w-pw)/2) : floor((w-pw)/2)+pw-1, :);
        
        % top-left crop
        I_stack(:, :, :, 5*(i-1)+2) = J(1:ph, 1:pw, :);
        
        % top-right crop
        I_stack(:, :, :, 5*(i-1)+3) = J(1:ph, w-pw+1:w, :);
        
        % bottom-left crop
        I_stack(:, :, :, 5*(i-1)+4) = J(h-ph+1:h, 1:pw, :);
        
        % bottom-right crop
        I_stack(:, :, :, 5*(i-1)+5) = J(h-ph+1:h, w-pw+1:w, :);
    end
end 
