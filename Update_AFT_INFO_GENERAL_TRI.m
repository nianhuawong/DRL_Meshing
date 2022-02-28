function AFT_stack_sorted = Update_AFT_INFO_GENERAL_TRI(AFT_stack_sorted, node1, node2, node3, nCells_AFT , xCoord_AFT, yCoord_AFT)
global cellNodeTopo;
cellNodeTopo(end+1,:) = [node1, node2, node3];

dist12 = DISTANCE(node1, node2, xCoord_AFT, yCoord_AFT);
dist13 = DISTANCE(node1, node3, xCoord_AFT, yCoord_AFT);
dist23 = DISTANCE(node2, node3, xCoord_AFT, yCoord_AFT);
%%
flag1 = IsLeftCell(node1, node2, node3, xCoord_AFT, yCoord_AFT);
[direction, row] = FrontExist(node1,node2, AFT_stack_sorted);    
if( flag1 == 1 )        %���Ϊ��Ԫ        
    if( row ~= -1 )  %����Ѿ�����
        if(direction == 1)  %���AFT_stack_sorted�д洢����Ϊ��node1��node2��
            AFT_stack_sorted(row, 3) = nCells_AFT;%����µ���Ԫ
        elseif( direction == -1 )
            AFT_stack_sorted(row, 4) = nCells_AFT; %������µ��ҵ�Ԫ
        end                                                   
    else %��������ڣ��򰴷����߼��������棬�����淽�򷴹�����node2, node1����nCells_AFT���µ��ҵ�Ԫ����Ԫ������
        AFT_stack_sorted(end+1,:) = [node2, node1, -1, nCells_AFT, dist12, size(AFT_stack_sorted,1)+1, 2];                     
    end
else  %���Ϊ�ҵ�Ԫ
    if( row ~= -1 ) %����Ѿ�����
        if(direction == 1) %���AFT_stack_sorted�д洢����Ϊ��node1��node2��
            AFT_stack_sorted(row, 4) = nCells_AFT; %����µ��ҵ�Ԫ
        elseif(direction == -1)
            AFT_stack_sorted(row, 3) = nCells_AFT; %������µ���Ԫ
        end                        
    else%��������ڣ��򰴷����߼��������棬�����淽�򷴹�����node2, node1����nCells_AFT���µ���Ԫ���ҵ�Ԫ������
        AFT_stack_sorted(end+1,:) = [node2, node1, nCells_AFT, -1, dist12, size(AFT_stack_sorted,1)+1, 2];                     
    end
end

flag2 = IsLeftCell(node2, node3, node1, xCoord_AFT, yCoord_AFT);
[direction, row] = FrontExist(node2,node3, AFT_stack_sorted); 
if( flag2 == 1 )
    if( row ~= -1 )
        if(direction == 1)
            AFT_stack_sorted(row, 3) = nCells_AFT;
        elseif( direction == -1 )
            AFT_stack_sorted(row, 4) = nCells_AFT;
        end  
    else
        AFT_stack_sorted(end+1,:) = [node3, node2, -1, nCells_AFT, dist23, size(AFT_stack_sorted,1)+1, 2];                     
    end                
else
    if( row ~= -1 )
        if(direction == 1)
            AFT_stack_sorted(row, 4) = nCells_AFT; 
        elseif(direction == -1)
            AFT_stack_sorted(row, 3) = nCells_AFT;
        end 
    else
        AFT_stack_sorted(end+1,:) = [node3, node2, nCells_AFT, -1, dist23, size(AFT_stack_sorted,1)+1, 2];                     
    end
end

flag3 = IsLeftCell(node3, node1, node2, xCoord_AFT, yCoord_AFT);
[direction, row] = FrontExist(node3,node1, AFT_stack_sorted); 
if( flag3 == 1 )
    if( row ~= -1 )
        if(direction == 1)
            AFT_stack_sorted(row, 3) = nCells_AFT;
        elseif( direction == -1 )
            AFT_stack_sorted(row, 4) = nCells_AFT;
        end  
    else
        AFT_stack_sorted(end+1,:) = [node1, node3, -1, nCells_AFT, dist13, size(AFT_stack_sorted,1)+1, 2];                     
    end                
else
    if( row ~= -1 )
        if(direction == 1)
            AFT_stack_sorted(row, 4) = nCells_AFT; 
        elseif(direction == -1)
            AFT_stack_sorted(row, 3) = nCells_AFT;
        end 
    else
        AFT_stack_sorted(end+1,:) = [node1, node3, nCells_AFT, -1, dist13, size(AFT_stack_sorted,1)+1, 2];                     
    end
end    