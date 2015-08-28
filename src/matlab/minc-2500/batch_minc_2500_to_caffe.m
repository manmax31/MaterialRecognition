% create train, test and validate.txt files in the Caffe format.
clear;

class_name_file = '/srv/datasets/Materials/OpenSurfaces/minc-2500/categories.txt';
m = class_name_2_idx_map(class_name_file);

minc_2500_label_dir = '/srv/datasets/Materials/OpenSurfaces/minc-2500/labels';
out_dir = '../../data/minc_2500';

subsets = {'train', 'test', 'validate'};
nSets = numel(subsets);
for s = 1:nSets
    for i = 1:5 % there are 5 tests 
        fprintf('Generate Caffe-formatted file for %s%d.txt \n', subsets{s}, i);        
        minc_file = sprintf('%s/%s%d.txt', minc_2500_label_dir, subsets{s}, i);
        caffe_file = sprintf('%s/%s%d.txt', out_dir, subsets{s}, i);
        minc_2_caffe(minc_file, m, caffe_file);
    end
end





