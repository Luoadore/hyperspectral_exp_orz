dataset = ["ksc", "ip", "sa", "pu"];
root = '/media/luo/result/hsi_ppr_result/';

i = 1;
path = root + dataset(i) + '/' + dataset(i) + '_0314_1_cube/data.mat';
load(path);

aa = zeros(16,1);


for i = 0:15

    index = (test_label == i);
    result = test_prediction(index);
    num = sum(index);
    correct = sum(result == i);
    aa(i+1,1) = correct / num;
end
t_oa = max(test_acc);