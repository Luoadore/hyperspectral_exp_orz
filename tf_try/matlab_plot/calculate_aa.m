load('/media/luo/result/hsi/extracted_data/SAdata.mat');
filedir = '/media/luo/result/hsi_kernels_r/SA/exp_';
aa = zeros(16,46);
for n = 1:46
    filename = [filedir, num2str(n+4), '\data.mat'];
    load(filename);
    for i = 0:15
        index = (test_label == i);
        result = test_prediction(index);
        num = sum(index);
        correct = sum(result == i);
        aa(i+1,n) = correct / num;
    end
end
%aa = aa';
%aa_mean = mean(aa);
%aa_std = std(aa);