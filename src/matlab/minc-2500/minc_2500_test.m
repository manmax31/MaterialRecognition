% Test a Caffe Net on a test dataset.
% 
% Input:
%     model: the .prototxt declaration of the test network architecture, 
%       including the data blob pointing to an LMDB file.    
% 
%     weights: the .caffemodel file containing the weights of a trained
%       network.
%       
%     n_test_batches: the number of image batches included in 
%       the whole test dataset.
%     
% Output:
%     acc: the average accuracy
%     loss: the average loss
% 
function [acc, loss] = minc_2500_test(model, weights, n_test_batches) 

    caffe.set_mode_gpu();
    caffe.set_device(0);

    test_net = caffe.Net(model, weights, 'test'); % create net for testing purposes

    n_net_outputs = length(test_net.outputs);
    test_res = zeros(n_test_batches, n_net_outputs);
    
    % Create cache of a sample image in each batch to make sure 
    % we have cycle through all of them.
    % shape = test_net.blobs('data').shape;
    % img_cache = zeros(shape(1), shape(2), shape(3), n_test_batches);
    
    % Loop through the LMDB test data a sufficient number of times
    % until whole test dataset is covered
    for i = 1:n_test_batches
        % Run test on the test samples preloaded in the model file.
        test_net.forward_prefilled();

        % Check input
        % data = test_net.blobs('data').get_data();
        % label = test_net.blobs('label').get_data();

        % To recover the first input image of the last test batch
        % img1 = data(:, :, :, 1); % 4D blobs coming from Caffe have dimensions
        % in the order of [width, height, channels, num], where width is
        % the fastest dimension.

        % swap width and height dimension
        % img_t = permute(img1, [2 1 3]);

        % add the mean per channel (specified during training) back
        % img_t(:, :, 1) = img_t(:, :, 1) + 104;
        % img_t(:, :, 2) = img_t(:, :, 2) + 117;
        % img_t(:, :, 3) = img_t(:, :, 3) + 124;
        % img_t = img_t(:, :, [3 2 1]); % swap from [B G R] channel order to [R G B] for display
        % imshow(uint8(img_t));
        
        % img_cache(:, :, :, mod(i-1, n_test_batches) + 1) = img_t;

        % Obtain intermediate layers' parameters
        % conv1 = test_net.blobs('conv1').get_data();
        % conv1_layer = test_net.layers('conv1');
        % conv1_weights = conv1_layer.params(1, 1).get_data();
        % conv1_bias = conv1_layer.params(1, 2).get_data();

        % Check output
        for j = 1:n_net_outputs
            test_res(i, j) = test_net.blobs(test_net.outputs{j}).get_data();
        end
    end

    % compute the average test accuracy and loss
    acc = mean(test_res(:, 1));
    loss = mean(test_res(:, 2));

    % clear the net to free memory
    clear test_net;
end
