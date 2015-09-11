% 
% Obtain the test accuracy of a Caffe Net on the (LMDB)
% test dataset assigned to it.
%
% Input:
%     test_net: the Caffe Net to be tested.
%     n_test_imgs: number of images (in an LMDB file) to be tested.
%
% Output:
%     acc: the average accuracy
%     loss: the average loss
%
function [acc, loss] = net_acc(test_net, n_test_imgs)

    data_size = test_net.blobs('data').shape;
    batch_size = data_size(4);

    % the number of image batches included in the whole test dataset.
    n_test_batches = ceil(n_test_imgs/batch_size);

    % number of net outputs (accuracy rate, loss, prob ...)
    n_net_outputs = length(test_net.outputs);
    test_res = zeros(n_test_batches, n_net_outputs);

    % Loop through the LMDB test data a sufficient number of times
    % until whole test dataset is covered
    for i = 1:n_test_batches
        
        % Run test on the test samples preloaded in the model file.
        test_net.forward_prefilled();

        % Check output
        for j = 1:n_net_outputs
            test_res(i, j) = test_net.blobs(test_net.outputs{j}).get_data();
        end
    end

    % compute the average test accuracy and loss 
    acc = mean(test_res(:, 1));
    loss = mean(test_res(:, 2));
end

