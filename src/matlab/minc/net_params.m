% Extract all the parameters of a network and concatenate them 
% into a long feature vector.
% 
% Input:
%   net: a Caffe Net object.
% 
% Output:
%   params: all the parameter values of all the (conv and fc) layers.
%   m: a Map object whose entry keys are layer names and values are structures 
%       containing the parameters in the corresponding layer.
function [params, m] = net_params(net, verbose)
    params = [];
        
    m = containers.Map;
        
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
end

