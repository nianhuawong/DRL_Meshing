%%
% myResetFunction('./boundary_file.cas')
function [InitialObservation,LoggedSignal] = myResetFunction(boundaryFile)
%% 初始状态为边界的最短阵面
[BC_stack, ~, ~, ~] = read_grid(boundaryFile, 0);
BC_stack_sorted = Sort_AFT(BC_stack);

base1 = BC_stack_sorted(1,1);
base2 = BC_stack_sorted(1,2);
PL = NeighborNodes(base1, BC_stack, base2);
PR = NeighborNodes(base2, BC_stack, base1);

LoggedSignal.State = [PL;base1;base2;PR];%只存点的编号
InitialObservation = LoggedSignal.State;
end