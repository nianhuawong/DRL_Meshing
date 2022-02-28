function flag = OutOfDomain(newpoint, RANGE)
x = newpoint(1);
y = newpoint(2);

Xmin = RANGE(1);
Xmax = RANGE(2);
Ymin = RANGE(3);
Ymax = RANGE(4);

flag = 1;
if x >= Xmin && x <= Xmax && y >= Ymin && y <= Ymax
    flag = 0;
end
end