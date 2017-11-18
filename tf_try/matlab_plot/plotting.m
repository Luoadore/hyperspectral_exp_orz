%Plot curve of accuracy and loss.

%root
train_root = 'F:\hsi_result\original\PU\lossAndaccuracy_train';
test_root = 'F:\hsi_result\original\PU\data';
root = 'F:\hsi_result\original\PU';

%load data
[steps, acc_1, loss_1] = read_train(train_root, 1);
[~, acc_4, loss_4] = read_train(train_root,1);
[~, acc_8, loss_8] = read_train(train_root, 8);
[steps_t, acc_1t, loss_1t] = read_test(test_root,1);
[~, acc_4t, loss_4t] = read_test(test_root,4);
[~, acc_8t, loss_8t] = read_test(test_root,8);

%plotting accuracy and save
plot(steps,acc_1,'b')
hold on
plot(steps_t,acc_1t,'r')
plot(steps,acc_4,'x-.')
plot(steps_t,acc_4t,'x-.')
plot(steps,acc_8,'s--')
plot(steps_t,acc_8t,'s--')
ylabel('accuracy')
xlabel('iterations')
title('Accuracy of original method with different neighbor(s) in PU')
legend('train_1','test_1','train_4','test_4','train_8','test_8')
saveas(gcf,strcat(root,'\acc_iters.fig'))
clf(gcf)

%plotting loss and save
plot(steps,loss_1,'b')
hold on
plot(steps_t,loss_1t,'r')
plot(steps,loss_4,'x-.')
plot(steps_t,loss_4t,'x-.')
plot(steps,loss_8,'s--')
plot(steps_t,loss_8t,'s--')
ylabel('loss')
xlabel('iterations')
title('Loss of original method with different neighbor(s) in PU')
legend('train_1','test_1','train_4','test_4','train_8','test_8')
saveas(gcf,strcat(root,'\loss_iters.fig'))
