% Convert a CNN to a fully convolutional network.
% 
% Input:
%   cnn_model_file: the .prototxt file declaring the original CNN architecture.
%   
% 	cnn_weight_file: the .caffemodel file containing the CNN weights.
% 
%   fc_layers: a cell array of strings indicating the names of the 
%       original FC layers to be converted.
% 
% 	fconvn_model_file: the .prototxt file declaring the fully conv net 
%       architecture to convert into.
% 
% 	fconvn_weight_file: the .caffemodel file where the params of the 
%       converted fully conv net are written to.
% 
%   fconv_layers: a cell array of the names of the corresponding 
%       fully conv layer converted from the original FC layers.
% 
% Output: 
%  	fcn: the output fully convolutional network object.
%   
function [fcn] = cnn2fconvn(cnn_model_file, cnn_weight_file, fc_layers, ...
        fconvn_model_file, fconvn_weight_file, fconv_layers)
    % input CNN (deploy mode)
    cnn = caffe.Net(cnn_model_file, cnn_weight_file, 'test');

    % output fully conv net
    fcn = caffe.Net(fconvn_model_file, cnn_weight_file, 'test');
    
    
    n_fc_layers = numel(fc_layers);   
    
    % For each layer
    for i = 1:n_fc_layers
        % load FC layer params from the CNN weight file.
        fc_weights = cnn.params(fc_layers{i}, 1).get_data();
        fc_bias    = cnn.params(fc_layers{i}, 2).get_data();
        
        % the shape of weight array of the converted fully conv layers
        fconv_weights_shape = fcn.params(fconv_layers{i}, 1).shape;
        
        % reshape to weights to an array of output x input x height x width 
        % dimensions that is compatible to the corresponding conv kernel
        fconv_weights = reshape(fc_weights, fconv_weights_shape);
        fcn.params(fconv_layers{i}, 1).set_data(fconv_weights);        
        
        % set the bias
        fcn.params(fconv_layers{i}, 2).set_data(fc_bias);
    end
    
    fcn.save(fconvn_weight_file);
end
