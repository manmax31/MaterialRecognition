clear;

I = imread('/usr/local/caffe/caffe-master/examples/images/cat.jpg');
p_size = [1; 227; 227];
scales = [1/sqrt(2); sqrt(2)];
[I_stack] = crop_and_scale(I, p_size, scales);

