%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test a converted fully conv network on an LMDB dataset by densely sliding 
% the input window of the network over the images.
% 
% Input:
%     test_net: the Caffe Net loaded with batch test data.
%     db_path : the path to the LMDB image database.
%     batch size: size of each image batch
% 
% Output:
%     acc: the recognition accuracy on the dataset
%     loss: the average loss per test sample
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [acc, conf_mat] = net_acc_fcn(test_net, db_path, batch_size)
    addpath('/home/huynh/Libs/matlab/matlab-lmdb');
    addpath('/home/huynh/Libs/matlab/encoder');

    database = lmdb.DB(db_path, 'RDONLY', true, 'NOLOCK', true);
    cursor = database.cursor('RDONLY', true);
    
    % set up crops and scales
    % input_size = test_net.blobs('data').shape; % test net's input size
    
%     scales = [1];
%     if ~ismember(1, scales)
%         scales = [1; scales];
%     end
    
    % number of samples per image (combinations of crops and scales)
    % n_samples = numel(scales);
    

    % batch_imgs = zeros(input_size(1), input_size(2), 3, batch_size);
    batch_gt_labels = zeros(batch_size, 1);
    
    % mean subtracted per channel
    channel_mean = [104, 117, 124]; 
            
    total_correct = 0; % total number of correct predictions    
    n_batches = 0;    
    n_test_imgs = 0; % the image count over the database
    
    % the confusion matrix
    prob = test_net.blobs('prob').get_data();
    n_classes = size(prob, 3);
    conf_mat = zeros(n_classes, n_classes);
    
    % a flag indicating if there is a next image in the database
    has_next = true;
    while (has_next)
        
        % linear index (wrt. the whole DB) of the first image in the batch
        batch_start = n_test_imgs + 1;

        % gather images and labels for the current batch
        for batch_img_idx = 1:batch_size % index images in the batch
            % advance to the next element in the DB, 
            % set the end_db flag to true if at the end of the DB.
            has_next = cursor.next();
            
            % if we reach the end of database, stop and perform
            % classification.
            if (~has_next)
                % update the batch size for the last one, which 
                % could be smaller than the previous ones 
                % (since it contains the left-over elements)
                batch_size = batch_img_idx - 1;
                break;
            else
                % decode LMDB entry into an (image, label) pair
                value = cursor.value;

                [img, label] = caffe.fromDatum(value);        
                img = imdecode(img, 'jpg');

                % Gather batch images and labels
                batch_imgs(:, :, :, batch_img_idx) = img;
                batch_gt_labels(batch_img_idx, 1) = label;

                % increase image count
                n_test_imgs = n_test_imgs+1;                
            end
        end
        
        % linear index (wrt. the whole DB) of the last image in the batch
        batch_end = n_test_imgs;
        
        % cut the batch 
        batch_imgs = batch_imgs(:, :, :, 1:batch_size);
        
        % Stop if no more images to process
        if (~has_next && batch_size == 0)
            break;
        end
        
        % perform batch classification once we gather enough images
        % for the batch OR reach the end of the DB
        n_batches = n_batches + 1;
        fprintf('Batch %d, ', n_batches);
        
        % convert to Caffe input format
        batch_imgs = mat_2_caffe_img(batch_imgs, channel_mean);

        % convert the ground truth class labels to be 1-based
        gt_labels(batch_start:batch_end, 1) = batch_gt_labels(1:batch_size, 1) + 1;

        % obtain the predicted class label from the max probability.
        test_net.blobs('data').reshape(size(batch_imgs)); % reshape net to input size
        res = test_net.forward({batch_imgs});
        
        % average (spatially) the class score for each image
        class_score = test_net.blobs('fc8-conv').get_data();
        img_class_score = reshape(mean(mean(class_score, 1), 2),  ...
            [n_classes, batch_size]); 
        
        % the predicted class is the one with the highest score
        [max_score, batch_pred_labels] = max(img_class_score, [], 1);
        pred_labels(batch_start:batch_end, 1) = batch_pred_labels;

        batch_correct = sum(batch_pred_labels' == gt_labels(batch_start:batch_end, 1));
        fprintf('correctly classified: %d/%d.\n', batch_correct, batch_size);

        % update the total number of correct predictions
        total_correct = total_correct + batch_correct;
    end
    clear cursor;
    clear database;
    
    % compute the average test accuracy and loss 
    acc = total_correct / n_test_imgs;
    
    % compute the conf_mat
    for i = 1:n_classes
        for j = 1:n_classes
            mask = (gt_labels == i) & (pred_labels == j);
            conf_mat(i, j) = sum(mask(:));
        end
    end
    
    class_counts = sum(conf_mat, 2); % number of samples in each class
    conf_mat = conf_mat ./ class_counts(:, ones(n_classes, 1));    
end

