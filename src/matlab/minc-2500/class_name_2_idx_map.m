% 
% This function reads a file storing a list of class names, one on each line. 
% and produces a map from class names as keys to their corresponding indices 
% 
% Input:
%   class_name_file: pathname to the file containing the class names.
% 
% Output:
%   m: the map storing (classname, class_idx) as key-value pairs.
%   Note that the class index is zero-based (to be consistent with 
%   the convention in Caffe.
% 
function m = class_name_2_idx_map(class_name_file)
    m = containers.Map;
    
    % read the class-name file    
    input_fid = fopen(class_name_file);

    % read the first class name
    mat_name = fgetl(input_fid);

    mat_idx = 0;
    
    while ischar(mat_name)        
        m(mat_name) = mat_idx; % look up material index
        
        % read next line
        mat_name = fgetl(input_fid);        
        mat_idx = mat_idx + 1;
    end
    
    % close file
    fclose(input_fid);
end
