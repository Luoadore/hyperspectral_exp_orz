function [steps,acc,loss] = read_train(root, neighbor)
%Load data set from csv, and return column array.
%
%   Args:
%        root: The root path.
%        neighbor: Pixel neighbor, include 1, 4, 8.
%
%    Return:
%        steps: Corresponding iterations.
%        acc: Accuracy of per 100 iterations.
%        loss: Loss of per 100 iterations.

acc_path = [root,'\run_',num2str(neighbor),'-tag-accuary.csv'];
loss_path = strcat(root,'\run_',num2str(neighbor),'-tag-training-loss.csv');
acc_data = csvread(acc_path,1,1);
loss_data = csvread(loss_path,1,1);
steps = acc_data(:,1);
acc = acc_data(:,2);
loss = loss_data(:, 2);

end

