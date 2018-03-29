filedir = 'E:\exp_result\exp_result\KSC\pred_';
aa = zeros(16,10);
for n = 1:10
    filename = [filedir, num2str(n), '.mat'];
    load(filename);
    for i = 0:15
        index = (test_label == i);
        result = test_prediction(index);
        num = sum(index);
        correct = sum(result == i);
        aa(i+1,n) = correct / num;
    end
end
aa = aa';
aa_mean = mean(aa);
aa_std = std(aa);
