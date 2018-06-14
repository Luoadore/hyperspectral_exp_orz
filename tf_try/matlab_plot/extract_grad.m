filedir = '/media/luo/cs/codestore/hyperspectral_exp_orz/tf_try/GANs/model/exp_11/data';
max_min = cell(10,2);
n = 1;
for i = 0:1000:9000
    filename = [filedir, num2str(i), '.mat'];
    load(filename);
    max_min{n, 1} = cal_max_min_grad(grad_real);
    max_min{n, 2} = cal_max_min_grad(grad_fake);
    n = n+1;
end