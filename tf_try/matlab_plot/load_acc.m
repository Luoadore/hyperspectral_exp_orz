filedir = 'E:\exp_result\IP\exp_';
max_acc = zeros(10,1);
max_test_acc = zeros(10,1);
for i= 1:10;
    filename = [filedir, num2str(i)];
    file = [filename, '\data.mat'];
    load(file);
    max_acc(i) = max(train_acc);
    max_test_acc(i) = max(test_acc);
end
acc_mean = mean(max_acc);
acc_test_mean = mean(max_test_acc);
acc_std = std(max_acc);
acc_test_std = std(max_test_acc);