plot_samples = zeros(10,176);
plot_labels = zeros(10,1);
for i = 1:10
    for j = 1:176
        plot_samples(i,j) = test_data(i, 5+(j-1)*9);
    end
    plot_labels(i)=test_label(i);
end
bands = 1:176;
plot(bands, plot_samples(1,:));
hold on
for i = 2:5
    plot(bands, plot_samples(i,:));
end
xlabel('value');
ylabel('bands');
%legend('class-1','class-10','class-11', 'class-2', 'class-0')