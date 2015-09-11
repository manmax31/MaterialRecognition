clear;

result_dir = '../../results/minc_2500/alexnet/28_08_2015';
max_epoch  = 50;

% % train with weight_decay = 0.05
% solver_file = 'minc_2500_alexnet_solver_26Aug2015_v1.prototxt';
% outfile_suffix = '26Aug2015_v1';
% minc_2500_train(solver_file, max_epoch, result_dir, outfile_suffix);
% 
% 
% % train with drop out ratio = 0.3
% solver_file = 'minc_2500_alexnet_solver_26Aug2015_v2.prototxt';
% outfile_suffix = '26Aug2015_v2';
% minc_2500_train(solver_file, max_epoch, result_dir, outfile_suffix);

% train with the decay factor of the learning rate of 0.5
% at a step of 4000 iterations.
% solver_file = 'minc_2500_alexnet_solver_27Aug2015_v1.prototxt';
% outfile_suffix = '27Aug2015_v1';
% minc_2500_train(solver_file, max_epoch, result_dir, outfile_suffix);

solver_file = 'minc_2500_alexnet_solver_28Aug2015_v1.prototxt';
outfile_suffix = '28Aug2015_v1';
minc_2500_train(solver_file, max_epoch, result_dir, outfile_suffix);