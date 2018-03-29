%load('')
plot(test_step, train_acc, 'x-.')
hold on
plot(test_step, test_acc, 'x-.')
xlabel('iterations')
ylabel('accuracy')
title('Accuracy of original method with 8-neighbors in SA')
legend('train', 'test')