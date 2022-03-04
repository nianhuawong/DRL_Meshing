function node = FindOneNeighborNode(node1_base, BC_stack, node2_base)
neighbors = NeighborNodes(node1_base, BC_stack, node2_base);

if(isempty(neighbors))
    error('neighbors is empty!');  
else
    node = neighbors(randi(length(neighbors)));
end
end