% load('/media/luo/result/hsi_gan_result/KSC/hsi_data11.mat')
bands = 1:176;
for i= 1:100
    % plot(bands, real(i, :))
    plot(bands, g_sample(i, :))
    hold on
end
title('11-class sample value')