load('/media/luo/result/hsi/extracted_data/KSCdata.mat');
load('/media/luo/result/hsi_ppr_result/ksc/ksc_0313/data.mat');
aa = zeros(13,1);
%a(n) = max(test_acc);
for i = 0:12
    index = (test_label == i);
    result = test_prediction(index);
    num = sum(index);
    correct = sum(result == i);
    aa(i+1,1) = correct / num;
end
t_a = max(test_acc);