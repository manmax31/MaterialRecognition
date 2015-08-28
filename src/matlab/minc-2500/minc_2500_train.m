% Train Caffe Net CNN on MINC-2500 test 1
% Input:
%   solver_file: the path to the solver file.
%   max_epoch: the total number of epochs 
%       (number of times that the training process 
%       sees all the training samples).
%   result_dir: the path to the result directory.
%   outfile_suffix: the suffix of all the output files.
% 
function minc_2500_train(solver_file, max_epoch, result_dir, outfile_suffix)

    % Perform training as well as monitoring a few parameters such as 
    % training and validation loss, and the ratios of parameter updates 
    % to parameter magnitudes.
    addpath(genpath('/usr/local/caffe/caffe-master/matlab'));

    caffe.set_mode_gpu();
    caffe.set_device(1);

    % solver configuration
    % solver_file = 'minc_2500_alexnet_solver_26Aug2015.prototxt';
    solver = caffe.Solver(solver_file);

    % Initialise weights from Alexnet for training
    alexnet_model = '/usr/local/caffe/caffe-master/models/bvlc_alexnet/bvlc_alexnet.caffemodel';
    train_net = solver.net;
    train_net.copy_from(alexnet_model);

    % Fine-tuning ...
    train_net = solver.net;
    val_net = solver.test_nets(1);

    % number of iterations per epoch
    train_data_size = train_net.blobs('data').shape; % training batch size
    train_batch = train_data_size(4);
    n_train = 48875; % number of training images. 

    val_data_size = val_net.blobs('data').shape; % training batch size
    val_batch = val_data_size(4);
    n_val = 2785; % number of validation images. 

    % the iteration at the beginning of the current epoch    
    cur_epoch_start = 0;

    % accuracy and loss
    train_acc = zeros(max_epoch, 1);
    train_loss = zeros(max_epoch, 1);
    val_acc = zeros(max_epoch, 1);
    val_loss = zeros(max_epoch, 1);

    min_param = zeros(max_epoch, 1);
    max_param = zeros(max_epoch, 1);
    min_param_update = zeros(max_epoch, 1);
    max_param_update = zeros(max_epoch, 1);

    prev_params = net_params(train_net);
    
    % track train and validation accuracy every epoch
    for i = 1:max_epoch    
        % the iteration at the end of the current  epoch
        cur_epoch_end = ceil(i * n_train/train_batch);

        % step until we finish an epoch
        solver.step(cur_epoch_end - cur_epoch_start);
        cur_epoch_start = cur_epoch_end;

        % now store the train and validation accuracy
        % [t_acc, t_loss] = net_acc(train_net, n_train);    
        train_acc(i) = train_net.blobs(train_net.outputs{1}).get_data();
        train_loss(i) = train_net.blobs(train_net.outputs{2}).get_data();

        [v_acc, v_loss] = net_acc(val_net, n_val);    
        val_acc(i) = v_acc;
        val_loss(i) = v_loss;

        % track the magnitudes of parameter updates 
        % and parameters.
        [params, m] = net_params(train_net);
        min_param(i) = min(params(:));
        max_param(i) = max(params(:));

        % compute param updates
        params_updates = params - prev_params;
        min_param_update(i) = min(params_updates(:));
        max_param_update(i) = max(params_updates(:));

        if (i == 1) % clean up the parameters created at the beginning
            clear prev_params;
        end

        % update the net. parameters for the current epoch
        prev_params = params;

        clear params;
        clear params_updates;
    end

    % Save train and validation history
    outfile = sprintf('%s/alexnet_train_val1_hist_%s.mat', result_dir, outfile_suffix);
    save(outfile, ...
        'train_acc', 'train_loss', 'val_acc', 'val_loss', ...
        'min_param', 'max_param', 'min_param_update', 'max_param_update');

    % Plot training and validation accuracy over epochs
    h = figure(1);
    plot(1:max_epoch, train_acc, 'r', 1:max_epoch, val_acc, 'b');
    legend('Training','Validation');
    xlabel('Epoch');
    title('Training and validation accuracy over epochs');
    acc_file = sprintf('%s/train_val_acc_%s.png', result_dir, outfile_suffix);
    print(h, '-dpng', acc_file);

    % Plot training and validation loss over epochs
    h = figure(2);
    plot(1:max_epoch, train_loss, 'r', 1:max_epoch, val_loss, 'b');
    legend('Training','Validation');
    xlabel('Epoch');
    title('Training and validation loss over epochs');
    loss_file = sprintf('%s/train_val_loss_%s.png', result_dir, outfile_suffix);
    print(h, '-dpng', loss_file);

    % Plot min and max parameters over epochs
    h = figure(3);
    plot(1:max_epoch, min_param, 'b', 1:max_epoch, max_param, 'r');
    legend('Min param','Max param');
    xlabel('Epoch');
    title('Min and max params over epochs');
    param_file = sprintf('%s/min_max_params_%s.png', result_dir, outfile_suffix);
    print(h, '-dpng', param_file);

    % Plot min and max param updates over epochs
    h = figure(4);
    plot(1:max_epoch, min_param_update, 'b', 1:max_epoch, max_param_update, 'r');
    legend('Min update', 'Max update');
    xlabel('Epoch');
    title('Min and max param updates over epochs');
    param_update_file = sprintf('%s/min_max_param_updates_%s.png', result_dir, outfile_suffix);
    print(h, '-dpng', param_update_file);

    % clear everything created by Caffe
    caffe.reset_all();
end