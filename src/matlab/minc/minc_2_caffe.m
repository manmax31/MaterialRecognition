% Convert the files containing pairs of image file paths and labels
% (e.g. train.txt, val.txt and test.txt files) in the format used in 
% the MINC-2500 dataset
% (http://opensurfaces.cs.cornell.edu/publications/minc/) 
% to the Caffe format.
% 
% The files in MINC contain several lines, each storing the class
% and path to the image file as follows.
% 
% images/brick/brick_002089.jpg
% where brick is the class name and images/brick/brick_002089.jpg 
% is the path to the image file.
% 
% Meanwhile, the format required by Caffe is as follows
% n01440764/n01440764_10066.JPEG 0
% where n01440764 is the class name and 
% n01440764/n01440764_10066.JPEG is the path to the image file
% and 0 is the index of the groundtruth class (n01440764).
% 
% Input: 
%   minc_file: the path to the input MINC file.
% 
%   m: the map storing (classname, class_idx) as key-value pairs.
%   Note that the class index is zero-based (to be consistent with 
%   the convention in Caffe.
% 
%   caffe_file: the path to the output Caffe file.
% 
function minc_2_caffe(minc_file, m, caffe_file)

    % Initialise a Map container with material names as keys 
    % and material indices as values.
    % class_name_file = '/srv/datasets/Materials/OpenSurfaces/minc-2500/categories.txt';
    % m = class_name_2_idx_map(class_name_file);

    % read the MINC file
    % minc_file = '../../data/minc/minc-2500/train1.txt';
    minc_fid = fopen(minc_file);

    % caffe_file = '../../data/minc/minc-2500/train1_caffe.txt';
    caffe_fid = fopen(caffe_file, 'w');

    % read a line
    tline = fgetl(minc_fid);

    while ischar(tline)
        % parse each line
        parsed = strsplit(tline, '/');
        mat_name = parsed{2}; % material name
        mat_idx = m(mat_name);% look up material index

        fprintf(caffe_fid, '%s %d\n', tline, mat_idx);

        % read next line
        tline = fgetl(minc_fid);
    end

    % closing files
    fclose(minc_fid);
    fclose(caffe_fid);
end
