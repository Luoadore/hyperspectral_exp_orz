function grad = cal_max_min_grad(grad_data)
% Calculate max
    grad_11 = grad_data{1, 1};
    grad_12 = grad_data{1, 2};
    grad_21 = grad_data{2, 1};
    grad_22 = grad_data{2, 2};
    grad_31 = grad_data{3, 1};
    grad_32 = grad_data{3, 2};
    grad = [max(max(grad_11)) min(min(grad_11))
        max(grad_12) min(grad_12)
        max(max(grad_21)) min(min(grad_21))
        max(grad_22) min(grad_22)
        max(grad_31) min(grad_31)
        max(grad_32) min(grad_32)];
end