classdef (Abstract) mesh_DRL_Abstract < rl.env.MATLABEnvironment
    properties
        nNodes     = 0;
        nFaces     = 0;
        nCells     = 0;
        Grid_stack = [];
        BC_stack   = [];
        boundaryFile =  './boundary_file.cas';
        Coord = [];
        Xmax = 0;
        Xmin = 0;
        Ymax = 0;
        Ymin = 0;
        RANGE = [];
    end
    
    properties
        % system state [L, B1, B2, R]'
        StateIndex = zeros(4,1)
    end
    
    properties(Access = protected)
        % Internal flag to store stale env that is finished
        IsDone = false
    end
    
    methods
        function this = mesh_DRL_Abstract(ActionInfo)
            %% 状态为输入模板点
            ObservationInfo = rlNumericSpec([4 2]);
            ObservationInfo.Name = 'mesh DRL States';
            ObservationInfo.Description = 'L, B1, B2, R';
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
        end
        
        function initialState = reset(this)
            [this.BC_stack, this.Coord, ~, ~] = read_grid(this.boundaryFile, 0);
            PLOT(this.BC_stack, this.Coord);
            
            this.Xmax = max(this.Coord(:,1));
            this.Ymax = max(this.Coord(:,2));
            this.Xmin = min(this.Coord(:,1));
            this.Ymin = min(this.Coord(:,2));
            this.RANGE = [this.Xmin, this.Xmax, this.Ymin, this.Ymax];
            
            BC_stack_sorted = Sort_AFT(this.BC_stack);
            
            node1_base = BC_stack_sorted(1,1);
            node2_base = BC_stack_sorted(1,2);
            PL = NeighborNodes(node1_base, this.BC_stack, node2_base);
            PR = NeighborNodes(node2_base, this.BC_stack, node1_base);

            initialState = [this.Coord(PL,:);this.Coord(node1_base,:);this.Coord(node2_base,:);this.Coord(PR,:)];
            this.StateIndex = [PL;node1_base;node2_base;PR];%只存点的编号
        end
        
        function [observation,reward,isdone,loggedSignals] = step(this,action)           
            loggedSignals = [];
            isdone = false;
            
            PL = this.StateIndex(1);
            node1_base = this.StateIndex(2);
            node2_base = this.StateIndex(3);
            PR = this.StateIndex(4);
            base_length = DISTANCE(node1_base, node2_base, this.Coord(:,1), this.Coord(:,2));
            
            %%  
            observation =[this.Coord(PL,:);this.Coord(node1_base,:);this.Coord(node2_base,:);this.Coord(PR,:)];
            loggedSignals.State = observation;

            %%
            Pbest = action';% action就是输出的新点坐标          
            plot(Pbest(1),Pbest(2),'bo');
            
            this.nFaces = size(this.BC_stack,1);
            %% 如果Pbest离base front很远，或者新点在区域外，则终止
            front_mid = 0.5 * (this.Coord(node1_base,:) + this.Coord(node2_base,:));
            VV = Pbest - front_mid;
            dist = sqrt(VV*VV');
            if dist > 10 * base_length || (OutOfDomain(Pbest, this.RANGE) == 1)
                isdone = true;
                this.IsDone = isdone;
                reward = -dist;    %距离越远，惩罚越大
                return;
            end

            %% Pbest在区域内部
            npoints = size(this.Coord, 1);
            node_select = npoints + 1;
            flag_best = 1;
            for i = 1:npoints
                Vec = this.Coord(i,:) - Pbest;
                dist1 = sqrt(Vec*Vec');
                if dist1 < base_length * 0.1     %如果输出的新点离现有点非常近，则选择现有点
                    Pbest = this.Coord(i,:);
                    node_select = i;
                    flag_best = 0;
                    break;
                end
            end
            
            if flag_best == 1   %如果选择了最佳点，则将最佳点坐标加入坐标点序列
                this.Coord(end+1,:) = Pbest;
            end
            %%
            nodeCandidate = NodeCandidate(this.BC_stack, node1_base, node2_base, this.Coord(:,1), this.Coord(:,2), Pbest, 3 * base_length);
            frontCandidate = FrontCandidate(this.BC_stack, nodeCandidate);
            flag_not_cross = IsNotCross(node1_base, node2_base, node_select, frontCandidate, this.BC_stack, this.Coord(:,1), this.Coord(:,2), 0);
            if flag_not_cross == 0
                isdone = true;
                this.IsDone = isdone;
                reward = -10;
                return;
            end
            %%
            [this.BC_stack, this.nCells] = UpdateTriCells(this.BC_stack, this.nCells, this.Coord(:,1), this.Coord(:,2), node_select, flag_best);
            
            num = size(this.BC_stack,1) - this.nFaces;
            PLOT_NEW_FRONT(this.BC_stack, this.Coord(:,1), this.Coord(:,2), num, flag_best)
            
            [this.BC_stack, this.Grid_stack] = DeleteInactiveFront(this.BC_stack, this.Grid_stack);
            
            quality = TriangleQuality(node1_base, node2_base, node_select, this.Coord(:,1), this.Coord(:,2));
            
            %%
            BC_stack_sorted = Sort_AFT(this.BC_stack);
            node1_base = BC_stack_sorted(1,1);
            node2_base = BC_stack_sorted(1,2);
            PL = NeighborNodes(node1_base, this.BC_stack, node2_base);
            PR = NeighborNodes(node2_base, this.BC_stack, node1_base);
            observation =[this.Coord(PL,:);this.Coord(node1_base,:);this.Coord(node2_base,:);this.Coord(PR,:)];                              
            loggedSignals.State = observation;
            
            this.StateIndex = [PL; node1_base; node2_base; PR];
            this.IsDone = isdone;
            %%
            if this.IsDone
                reward = -10;
            else
                reward = quality;
            end
            
            %% 考虑加上剩余面积判断，填满计算域则给奖励1
%             if this.nCells == 5
%                 reward = ;
%             end
        end
    end
end