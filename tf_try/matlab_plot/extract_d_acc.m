a = zeros(111,2);
d_acc = d_acc';
for i=1:111
    a(i,1) = d_acc{i}(1);
    a(i,2) = 1 - d_acc{i}(2);
end
