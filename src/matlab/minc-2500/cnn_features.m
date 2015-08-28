% Extract CNN features at all the layers from a given image.
% The input image size (height and width) does not need to match the input
% size of data layer of the CNN.
% 
% Input:
%   net: the Caffe net object representing a CNN.
%   img: the input (colour or gray) image.
% 
% Output:
%   f: a Map object containing (key, value) entry pairs, where the keys are 
%       layer names and each value is a 3D array storing the feature
%       vectors generated as tbe output of the corresponding layer.
% 
function [f] = cnn_features(net, img)
  
    % convert img into a cell array accepted by the net

    % load the image into the net.
    res = net.forward({data});

    f = containers.Map;
    
    n_layers = numel(net.layer_names);
    
    if (nargin == 1) 
        % if not specified, no verbose output by default
        verbose = false; 
    end
    
    for i = 1:n_layers
        layer = net.layers(net.layer_names{i}); 
        layer_type = layer.type;

        % get parameters
        if (strcmp(layer_type, 'Convolution') || strcmp(layer_type, 'InnerProduct'))
            weights = layer.params(1).get_data();
            bias    = layer.params(2).get_data();
            
            % put all the parameters for each layer in a separate bucket 
            layer_params.weights = weights(:);
            layer_params.bias = bias(:);
            m(net.layer_names{i}) = layer_params;
            params = [params; weights(:); bias(:)];
            
            % print variance of each layer
            if (verbose)
                fprintf('Layer % 8s: weights (mean = % 8.5f, std = % 8.5f), bias (mean = % 8.5f, std = % 8.5f)\n', ...
                net.layer_names{i}, mean(weights(:)), std(weights(:)), mean(bias(:)), std(bias(:)));
            end
        end
    end     

ends