function [] = plot_acc(dataset)
    %load('')
    plot(test_step, train_acc, 'x-.')
    hold on
    plot(test_step, test_acc, 'x-.')
    xlabel('iterations')
    ylabel('accuracy')
    %title('Accuracy of original method with 8-neighbors in KSC')
    title(['Accuracy of PPR in ', dataset])
    legend('train', 'test')
    saveas(gcf, ['/media/luo/cs/ppr_', dataset, '_acc.png'])
end