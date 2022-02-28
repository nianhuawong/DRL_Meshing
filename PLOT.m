function PLOT(AFT_stack, Coord)
xCoord = Coord(:,1);
yCoord = Coord(:,2);

fig = gcf;
clf;

fig.Color = 'white'; hold on;
len = size(AFT_stack,1);
for i = 1:len
    node1 = AFT_stack(i,1);
    node2 = AFT_stack(i,2);
        
    xx = [xCoord(node1),xCoord(node2)];
    yy = [yCoord(node1),yCoord(node2)];

%     plot( xx, yy, '-r.', 'MarkerSize',14);
    plot( xx, yy, '-r','LineWidth',1);
    hold on;
end
axis equal;
axis off;

end

% global flag_label num_label;
% nodeList = AFT_stack(:,1:2);
% nNodes = max( max(nodeList)-min(nodeList)+1 );
% for i = 1 : nNodes
%     str = num2str(i);
%     if  flag_label(i) == 0 && num_label == 1
%         text(xCoord(i)+0.00005*dist,yCoord(i)+0.00005*dist,str, 'Color', 'red', 'FontSize', 9)
%         flag_label(i) = 1;
%     end
% end

    
