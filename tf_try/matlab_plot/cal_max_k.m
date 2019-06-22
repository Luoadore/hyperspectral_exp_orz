cls=cell(13,2);
for i=1:13
    a=aa(i,:);
    index=find(a == max(a));
    cls(i,1) = {max(a)};
    cls(i,2) = {index + 4};
end
