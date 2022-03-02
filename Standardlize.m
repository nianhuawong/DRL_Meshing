function input = Standardlize(input)
% point1 = [input(1), input(5)];
% point2 = [input(2), input(6)];
% point3 = [input(3), input(7)];
% point4 = [input(4), input(8)];
point1 = input(1,:);
point2 = input(2,:);
point3 = input(3,:);
point4 = input(4,:);
[input(1,1), input(1,2)] = Transform( point1, point2, point3 );
input(2,1) = 0.0; input(2,2) = 0;
input(3,1) = 1.0; input(3,2) = 0;
[input(4,1), input(4,2)] = Transform( point4, point2, point3 );
end