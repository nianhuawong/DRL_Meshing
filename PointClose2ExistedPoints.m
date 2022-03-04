function index = PointClose2ExistedPoints(Point,Coord,node1_base,node2_base)
index = -1;
coeff = 0.5;

base_length = DISTANCE(node1_base, node2_base, Coord(:,1), Coord(:,2));
mindis = 1e40;
for i = 1:size(Coord, 1)
    Vec = Coord(i,:) - Point;
    dist = sqrt( Vec * Vec' );
    if dist < base_length * coeff && i~= node1_base && i~= node2_base
        if dist < mindis
            mindis = dist;
            index = i;
        end
    end
end

end