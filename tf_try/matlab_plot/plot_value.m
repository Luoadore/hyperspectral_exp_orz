dir = 'C:\Users\Mayday\Desktop\model£¨6-18£©\exp_11\data';
bands = 1:176;
for n = 0: 1000:9000
    filename = [dir, num2str(n), '.mat'];
    load(filename);
    for i= 1:100
        %plot(bands, data(i, :))
        plot(bands, g_sample(i, :))
        hold on
    end
    title(['11-class sample value-', num2str(n)])
    %title('11-class value')
    saveas(gcf, ['g_sample', num2str(n), '.jpg'])
end