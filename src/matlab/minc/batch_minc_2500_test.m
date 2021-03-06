clear;

addpath(genpath('/usr/local/caffe/caffe-master/matlab'));

caffe.set_mode_gpu();
caffe.set_device(1);

% Test all the snapshot models (stored at every 10000 iterations) 
% on the test dataset.
model = 'minc_2500_alexnet_test.prototxt';

% the number of batches of test images
n_test_imgs = 5750;

n_classes = 23; % number of classes

% the number of iterations between two successive snapshots.
snapshot_step = 2000; 
% snapshot_iters = [500:snapshot_step:2000]; 
% snapshot_iters = [10000:snapshot_step:450000];
snapshot_iters = [2000:snapshot_step:38000];
n_snapshots = numel(snapshot_iters);

acc = zeros(n_snapshots , 1); % test acc per trained net.
loss = zeros(n_snapshots , 1); % loss per trained net.
mean_class_acc = zeros(n_snapshots , 1); % mean class accuracy per trained net.
conf_mat = zeros(n_classes, n_classes, n_snapshots); % confusion matrix obtained by each trained net.

test_net = caffe.Net(model, 'test'); % create net for testing purposes

iter_idx = 0;
for iter = snapshot_iters
    fprintf('Test snapshot model at %d iterations ...\n', iter);
    iter_idx = iter_idx + 1;
    weights = sprintf('../../results/minc_2500/alexnet/27_08_2015/minc_2500_alexnet_train1_27Aug2015_v1_iter_%d.caffemodel', iter);
 
    % [test_acc, test_loss] = minc_2500_test(model, weights, n_test_batches);    
    test_net.copy_from(weights);
    % [test_acc, test_loss] = net_acc(test_net, n_test_imgs);
    [test_acc, test_loss, test_conf_mat] = net_acc_conf_mat(test_net, n_test_imgs);
    test_mean_class_acc = mean(diag(test_conf_mat));
    
    acc(iter_idx, 1) = test_acc; 
    loss(iter_idx, 1) = test_loss; 
    conf_mat(:, :, iter_idx) = test_conf_mat;
    mean_class_acc(iter_idx, 1) = test_mean_class_acc;
end
    
% % save('../../results/minc_2500/alexnet/12_08_2015/alexnet_test1_acc.mat',  'snapshot_iters', 'acc', 'loss');
% % save('../../results/minc_2500/alexnet/23_08_2015/alexnet_test1_acc.mat', 'snapshot_iters', 'acc', 'loss');
% % save('../../results/minc_2500/alexnet/23_08_2015/alexnet_test1_acc_iters_500_2000.mat', 'snapshot_iters', 'acc', 'loss');
% % save('../../results/minc_2500/alexnet/23_08_2015/alexnet_test1_acc_fc8_tuned.mat', 'snapshot_iters', 'acc', 'loss');
% % save('../../results/minc_2500/alexnet/28_08_2015/alexnet_test1_iters_2000_38000.mat', 'snapshot_iters', 'acc', 'loss');
save('../../results/minc_2500/alexnet/27_08_2015/alexnet_test1_iters_2000_38000.mat', 'snapshot_iters', 'acc', 'loss', 'conf_mat', 'mean_class_acc', '-append');
% save('../../results/minc_2500/alexnet/28_08_2015/minc_alexnet_Bell_etal.mat', 'acc', 'loss');


% Plot classification results
h = figure(1);
plot(acc);
xlabel('Iterations (x2000)');
title('Test accuracy over snapshot iterations');
% print(h, '-dpng', '../../results/minc_2500/alexnet/12_08_2015/test1_acc.png');
% print(h, '-dpng', '../../results/minc_2500/alexnet/23_08_2015/test1_acc.png');
% print(h, '-dpng', '../../results/minc_2500/alexnet/23_08_2015/test1_acc_iters_500_2000.png');
% print(h, '-dpng', '../../results/minc_2500/alexnet/23_08_2015/test1_acc_fc8_tuned.png');
print(h, '-dpng', '../../results/minc_2500/alexnet/28_08_2015/test1_acc_iters_2000_38000.png');

% Plot loss on the test set
h = figure(2);
plot(loss);
xlabel('Iterations (x2000)');
title('Loss over snapshot iterations');
% print(h, '-dpng', '../../results/minc_2500/alexnet/12_08_2015/test1_loss.png');
% print(h, '-dpng', '../../results/minc_2500/alexnet/23_08_2015/test1_loss.png');
% print(h, '-dpng', '../../results/minc_2500/alexnet/23_08_2015/test1_loss_iters_500_2000.png');
print(h, '-dpng', '../../results/minc_2500/alexnet/28_08_2015/test1_loss_iters_2000_38000.png');

% Plot the mean class accuracy
h = figure(3);
plot(mean_class_acc);
xlabel('Iterations (x2000)');
title('Mean class accuracy over snapshot iterations');
print(h, '-dpng', '../../results/minc_2500/alexnet/28_08_2015/test1_mean_class_acc_iters_2000_38000.png');

% clear everything created by Caffe
clear test_net;
caffe.reset_all();
