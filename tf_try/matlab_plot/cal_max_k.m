cls=cell(16,2);
for i=1:16
    a=aa(i,:);
    index=find(a == max(a));
    cls(i,1) = {max(a)};
    cls(i,2) = {index + 4};
end
