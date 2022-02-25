function env = mesh_DRL_env(boundaryFile)
nCells     = 0; 
Grid_stack = [];
%%
[BC_stack, Coord, ~, ~] = read_grid(boundaryFile, 0);
PLOT(BC_stack, Coord);

%% 状态为输入模板点
ObservationInfo = rlNumericSpec([4 1]);
ObservationInfo.Name = 'mesh DRL States';
ObservationInfo.Description = 'L, B1, B2, R';

%% 动作为生成一个新点坐标
ActionInfo = rlNumericSpec([1 2]);
ActionInfo.Name = 'mesh DRL Action';
ResetHandle = @() myResetFunction(boundaryFile);
StepHandle = @(Action,LoggedSignals) myStepFunction(Action,LoggedSignals, BC_stack, Coord, nCells, Grid_stack);

env = rlFunctionEnv(ObservationInfo,ActionInfo,StepHandle,ResetHandle);

end

