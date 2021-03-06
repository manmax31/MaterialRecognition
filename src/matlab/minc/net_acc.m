%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Obtain the test accuracy of a Caffe Net on an LMDB test dataset 
% that has been fed into a test network (in the network prototxt file).
%
% Input:
%     test_net: the Caffe Net loaded with batch test data.
%     n_test_imgs: number of images (in an LMDB file) to be tested.
%
% Output:
%     acc: the overall recognition accuracy on the dataset.
%     loss: the average loss per test sample
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [acc, loss] = net_acc(test_net, n_test_imgs)
    data_size = test_net.blobs('data').shape;
    batch_size = data_size(4);

    % the number of image batches included in the whole test dataset.
    n_test_batches = ceil(n_test_imgs/batch_size);
    
    total_correct = 0; % total number of correct predictions
    total_loss    = 0; % total loss over all the test samples
    
    % Loop through the LMDB test data a sufficient number of times
    % until whole test dataset is covered
    for i = 1:n_test_batches
        fprintf('Batch %d/%d, ', i, n_test_batches);
        
        % run test on the test samples preloaded with the net object.
        test_net.forward_prefilled();

        % for the last batch, the size could be smaller than the previous
        % one (containing the left-overs)
        if (i == n_test_batches)
            batch_size = n_test_imgs - (i-1) * batch_size;
        end
        
        % compute the number of correct predictions 
        % and average loss for the current batch        
        batch_correct = round(test_net.blobs('accuracy').get_data() * batch_size);
        avg_batch_loss = test_net.blobs('loss').get_data();
        
        fprintf('correct: %d/%d, ', batch_correct, batch_size);
        fprintf('loss per batch sample: %g\n', avg_batch_loss);
        
        % update the total number of correct predictions
        total_correct = total_correct + batch_correct;
        total_loss = total_loss + avg_batch_loss * batch_size;
    end

    % compute the average test accuracy and loss 
    acc = total_correct / n_test_imgs;
    loss = total_loss / n_test_imgs;
end
