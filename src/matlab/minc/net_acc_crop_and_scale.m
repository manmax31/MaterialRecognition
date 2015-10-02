%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test a trained CNN network on an LMDB dataset by cropping different 
% portions of the image at different scales.
% 
% Input:
%     test_net: the Caffe Net loaded with batch test data.
%     db_path : the path to the LMDB image database.
% 
% Output:
%     acc: the recognition accuracy on the dataset
%     loss: the average loss per test sample
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [acc, loss] = net_acc_crop_and_scale(test_net, db_path)
    addpath('/home/huynh/Libs/matlab/matlab-lmdb');
    addpath('/home/huynh/Libs/matlab/encoder');

    database = lmdb.DB(db_path, 'RDONLY', true, 'NOLOCK', true);
    cursor = database.cursor('RDONLY', true);
    
    % a cursor pointing to the end of the database
    last_cursor = database.cursor('RDONLY', true);
    last_cursor.last();
    last_key = last_cursor.key();
    
    % set up crops and scales
    input_size = test_net.blobs('data').shape; % test net's input size
    p_size = [input_size(1); input_size(2)];
    scales = [0.5; 1/sqrt(2)];
    if ~ismember(1, scales)
        scales = [1; scales];
    end
    
    % number of samples per image (combinations of crops and scales)
    n_samples = numel(scales) * 5;
    
    channel_mean = [104, 117, 124]; % mean subtracted per channel
    
    % batch size
    batch_size = 23; % 8 images x 15 samples/image = 120 samples/batch
     
    batch_imgs = zeros(input_size(1), input_size(2), 3, n_samples * batch_size);
    batch_labels = zeros(batch_size, 1);
        
    total_correct = 0; % total number of correct predictions    
    n_batches = 0;    
    n_test_imgs = 0; % the image count over the database
    while cursor.next()
        key = cursor.key;
        
        % decode LMDB entry into an (image, label) pair
        value = cursor.value;
        
        [img, label] = caffe.fromDatum(value);        
        % [label, image] = caffe.caffe_('from_datum', value);
        % [img, label] = caffe_proto_('fromDatum', value);
        % img = imdecode(img);
        
        % crop and scale each image        
        [img_stack] = crop_and_scale(img, p_size, scales);
        
        % number of samples per image (combinations of crops and scales)
        % n_samples = size(img_stack, 4); 
        
        % Gather batch images and labels
        batch_img_idx = mod(n_test_imgs, batch_size)+1; % image index in the current batch
        batch_imgs(:, :, :, n_samples*(batch_img_idx-1)+1:n_samples*batch_img_idx) = img_stack;
        batch_labels(batch_img_idx, 1) = label;
        
        % increase image count
        n_test_imgs = n_test_imgs+1;
        
        % perform batch classification once have enough
        % OR reach the end of the DB
        if ((batch_img_idx == batch_size) || isequal(key, last_key))
            n_batches = n_batches + 1;
            fprintf('Batch %d, ', n_batches);
            
            % convert to Caffe input format
            batch_imgs = mat_2_caffe_img(batch_imgs, channel_mean, ...
            [input_size(1), input_size(2)]);
            
            test_net.blobs('data').reshape(size(batch_imgs));
            % test_net.reshape();             
            
            % obtain the predicted class label from the max probability.
            res = test_net.forward({batch_imgs});
            prob = res{1};
            n_classes = size(prob, 1);
            prob = reshape(prob, [n_classes, n_samples, batch_size]); % group samples for each image
            mean_prob = mean(prob, 2); % average the class prob across samples in each image
            mean_prob = reshape(mean_prob, [n_classes, batch_size]);
            [max_prob, max_class_idx] = max(mean_prob, [], 1);

            % number of correct predictions for the current batch
            % (convert the class labels to 0-based)
            batch_correct = sum(max_class_idx' - 1 == batch_labels);
            fprintf('correctly classified: %d/%d.\n', batch_correct, batch_size);
            
             % update the total number of correct predictions
            total_correct = total_correct + batch_correct;
        end
    end
    clear cursor;
    clear last_cursor;
    clear database;
    
    % compute the average test accuracy and loss 
    acc = total_correct / n_test_imgs;
end

