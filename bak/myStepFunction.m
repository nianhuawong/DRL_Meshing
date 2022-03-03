function [NextObs,Reward,IsDone,LoggedSignals] =  myStepFunction(Action,LoggedSignals, BC_stack, Coord, nCells, Grid_stack)
%%
newpoint = Action;
State = LoggedSignals.State;
PL = State(1);
node1_base = State(2);
node2_base = State(3);
PR = State(4);
base_length = DISTANCE(node1_base, node2_base, Coord(:,1), Coord(:,2));

npoints = size(Coord, 1);
node_select = npoints + 1;
flag_best = 1;
for i = 1:npoints
    VV = Coord(i,:) - newpoint;
    dist = sqrt(VV.*VV);
    if dist < base_length * 0.1
        newpoint = Coord(i,:);
        node_select = i;
        flag_best = 0;
        break;
    end
end

if flag_best == 1
    Coord(end+1,:) = newpoint;
end
%%
[BC_stack, nCells] = UpdateTriCells(BC_stack, nCells, Coord(:,1), Coord(:,2), node_select, flag_best);

[BC_stack, Grid_stack] = DeleteInactiveFront(BC_stack, Grid_stack);

%%
BC_stack_sorted = Sort_AFT(BC_stack);

node1_base = BC_stack_sorted(1,1);
node2_base = BC_stack_sorted(1,2);
PL = NeighborNodes(node1_base, BC_stack, node2_base);
PR = NeighborNodes(node2_base, BC_stack, node1_base);
LoggedSignals.State = [PL, node1_base, node2_base, PR];

NextObs = LoggedSignals.State;

IsDone = false;
%%
nodeCandidate = NodeCandidate(BC_stack, node1_base, node2_base, Coord(:,1), Coord(:,2), newpoint, 3 * base_length);
frontCandidate = FrontCandidate(BC_stack, nodeCandidate);
flag_not_cross = IsNotCross(node1_base, node2_base, newpoint, frontCandidate, BC_stack, Coord(:,1), Coord(:,2), 0);
if flag_not_cross == 0
    IsDone = true;
end

%%
quality = TriangleQuality(node1_base, node2_base, newpoint_Index, Coord(:,1), Coord(:,2));
if IsDone
    Reward = -10;
else
    Reward = quality;
end
end

function quality = TriangleQuality(node1, node2, node3, xCoord, yCoord)

a = DISTANCE( node1, node2, xCoord, yCoord) + 1e-40;  
b = DISTANCE( node2, node3, xCoord, yCoord) + 1e-40;       
c = DISTANCE( node3, node1, xCoord, yCoord) + 1e-40;

tmp = ( a^2 + b^2 - c^2 ) / ( 2.0 * a * b );
if abs(tmp-1.0)<1e-5
    tmp = 1;
end

if abs(tmp+1.0) < 1e-5
    tmp = -1;
end

theta = acos( tmp );

area = 0.5 * a * b * sin(theta) + 1e-40;             %三角形面积
% r = 2.0 * area / ( ( a + b + c ) );                  %内切圆半径
% R = a * b * c / 4.0 / area + 1e-40;                  %外接圆半径  
% 
% quality =   3.0 * r / R;

%% 三角形网格质量的另一种求法
quality = 4.0 * sqrt(3.0) * area / ( a * a + b * b + c * c );
end