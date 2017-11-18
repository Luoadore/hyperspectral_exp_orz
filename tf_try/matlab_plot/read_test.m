function [steps, acc, loss] = read_test(root, neighbor)
%Load data set from mat, and return column array.
%
%   Args:
%        root: The root path.
%        neighbor: Pixel neighbor, include 1, 4, 8.
%
%    Return:
%        steps: Corresponding iterations.
%        acc: Accuracy of per 100 iterations.
%        loss: Loss of per 100 iterations.

path = [root,'\data',num2str(neighbor),'.mat'];
load(path);
steps = test_step';
acc = test_acc';
loss = test_loss';

end

