%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Obtain the test accuracy of a Caffe Net on an LMDB test dataset 
% that has been fed into a test network (in the network prototxt file).
% The output also includes the class confusion matrix.
% Input:
%     test_net: the Caffe Net loaded with batch test data.
%     n_test_imgs: number of images (in an LMDB file) to be tested.
%
% Output:
%     acc: the overall recognition accuracy on the dataset.
%     loss: the average loss per test sample
%     conf_mat: a confusion matrix, where the row corresponds to the ground
%     truth classes and columns corresponding to the predicted classes. The
%     matrix size is n_classes x n_classes.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [acc, loss, conf_mat] = net_acc_conf_mat(test_net, n_test_imgs)
    data_size = test_net.blobs('data').shape;
    batch_size = data_size(4);

    % the number of image batches included in the whole test dataset.
    n_test_batches = ceil(n_test_imgs/batch_size);
    
    total_correct = 0; % total number of correct predictions
    total_loss    = 0; % total loss over all the test samples
   
    gt_labels = zeros(n_test_imgs, 1);
    pred_labels = zeros(n_test_imgs, 1);
    
    % Loop through the LMDB test data a sufficient number of times
    % until whole test dataset is covered
    for i = 1:n_test_batches
        fprintf('Batch %d/%d, ', i, n_test_batches);
        
        % for the last batch, the size could be smaller than the previous
        % one (containing the left-overs)
        % linear index of the first image in the batch
        batch_start = (i-1)*batch_size + 1; 
        if (i < n_test_batches)
            % linear index of the last image in the batch
            batch_end = i*batch_size;
        else
            batch_end = n_test_imgs;
            batch_size = batch_end - batch_start + 1;
        end
        
        % run test on the test samples preloaded with the net object.
        test_net.forward_prefilled();

        % convert the groundtruth class labels to be 1-based
        label_buffer = test_net.blobs('label').get_data();
        gt_labels(batch_start:batch_end, 1) = label_buffer(1:batch_size, 1) + 1;
        
        % compute the number of correct predictions
        % and average loss for the current batch        
        batch_correct = round(test_net.blobs('accuracy').get_data() * batch_size);
        avg_batch_loss = test_net.blobs('loss').get_data();
        
        % the predicted labels for the current batch
        batch_prob = test_net.blobs('prob').get_data();
        [max_prob, batch_pred_class] = max(batch_prob, [], 1);        
        pred_labels(batch_start:batch_end, 1) = batch_pred_class(1:batch_size)'; 
        
        fprintf('correct: %d/%d, ', batch_correct, batch_size);
        fprintf('loss per batch sample: %g\n', avg_batch_loss);
        
        % update the total number of correct predictions
        total_correct = total_correct + batch_correct;
        total_loss = total_loss + avg_batch_loss * batch_size;
    end

    % compute the average test accuracy and loss 
    acc = total_correct / n_test_imgs;
    loss = total_loss / n_test_imgs;
    
    % compute the confusion matrix
    n_classes = max(gt_labels);
    conf_mat = zeros(n_classes, n_classes);
    for i = 1:n_classes
        for j = 1:n_classes
            mask = (gt_labels == i) & (pred_labels == j);
            conf_mat(i, j) = sum(mask(:));    
        end
    end
    
    class_counts = sum(conf_mat, 2); % number of samples in each class
    conf_mat = conf_mat ./ class_counts(:, ones(n_classes, 1));
end
