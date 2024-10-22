function  [AFT_stack, nCells_AFT] = UpdateTriCells(AFT_stack, nCells_AFT, xCoord_AFT, yCoord_AFT, node_select, flag_best)
epsilon = 0.8;
%阵面及左单元编号更新
node1_base = AFT_stack(1,1);         %阵面的基准点
node2_base = AFT_stack(1,2);

nCells_AFT = nCells_AFT + 1;
AFT_stack = Update_AFT_INFO_TRI(AFT_stack, node1_base, ...
    node2_base, node_select,nCells_AFT, xCoord_AFT, yCoord_AFT);
%%
%新增点和阵面之后，除了基准阵面形成的单元，还有可能会自动构成多个单元，需要判断。
%在已有阵面中查找自动构成的单元，判断方法为邻点的邻点如果相邻，则构成封闭单元
%新增点的邻点
if flag_best == 0
    neighbor = NeighborNodes(node_select, AFT_stack,-1);%找出现有点的相邻点
    neighbor = [node1_base, node2_base, neighbor];
    neighbor = unique(neighbor);
           
    %         邻点中若是有互相相邻的，则形成新单元
    new_cell = [];
    for i = 1:length(neighbor)
        neighborNode = neighbor(i);
        for j = 1: size(AFT_stack,1)
            if( AFT_stack(j,1) == neighborNode ) %邻点所在阵面
                if( find(neighbor==AFT_stack(j,2)) )  %邻点的邻点如果
                    new_cell(end+1,:) = [node_select, neighborNode, AFT_stack(j,2)];
                end
            end
            if(AFT_stack(j,2) == neighborNode)
                if( find(neighbor==AFT_stack(j,1)) )
                    new_cell(end+1,:) = [node_select, neighborNode, AFT_stack(j,1)];
                end
            end
        end
    end
    
    %去掉重复的单元
    for i = 1 : size(new_cell,1)
        iCell = new_cell(i,:);
        for j = i+1:size(new_cell,1)
            jCell = new_cell(j,:);
            if( sum(jCell([3,2]) == iCell([2,3])) == 2)
                new_cell(j,:)=-1;
            end
        end
    end
    
    II = new_cell(:,1) == -1;
    new_cell(II,:)=[];
    
    %还要去掉已经有的单元
    for i = 1 : size(new_cell,1)
        iCell = new_cell(i,:);
        if(size(find(iCell==node1_base),2) && size(find(iCell==node2_base),2) && size(find(iCell==node_select),2))
            new_cell(i,:)=-1;
        end
    end
    
    II = new_cell(:,1) == -1;
    new_cell(II,:)=[];
    
    %还要去掉质量不好的单元
    for i = 1 : size(new_cell,1)
        node1 = new_cell(i,1);
        node2 = new_cell(i,2);
        node3 = new_cell(i,3);
        
        [quality,~] = QualityCheckTri(node1, node2, node3, xCoord_AFT, yCoord_AFT, -1);
%         if abs( 1.5-quality ) > epsilon
        if quality < epsilon
            new_cell(i,:)=-1;
        end
    end
    II = new_cell(:,1) == -1;
    new_cell(II,:)=[];
    
    %将新单元加入数据结构
    for i = 1:size(new_cell,1)
        nCells_AFT = nCells_AFT + 1;
        node1 = new_cell(i,1);
        node2 = new_cell(i,2);
        node3 = new_cell(i,3);
        
        AFT_stack = Update_AFT_INFO_GENERAL_TRI(AFT_stack, ...
            node1, node2, node3, nCells_AFT , xCoord_AFT, yCoord_AFT);
    end
end

end