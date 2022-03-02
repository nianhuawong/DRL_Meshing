classdef (Abstract) mesh_DRL_Abstract < rl.env.MATLABEnvironment
    properties
        nNodes     = 0;
        nFaces     = 0;
        nCells     = 0;
        Grid_stack = [];
        BC_stack0  = [];
        BC_stack   = [];
        boundaryFile =  './boundary_file.cas';
        Coord0 = [];
        Coord  = [];
        Xmax = 0;
        Xmin = 0;
        Ymax = 0;
        Ymin = 0;
        RANGE = [];
        PenaltyForOutOfDomain = -10;
        PenaltyForCross = -10;
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
            %% 状态为输入模板点的x,y坐标
            ObservationInfo = rlNumericSpec([4 2]);
            ObservationInfo.Name = 'mesh DRL States';
            ObservationInfo.Description = 'L, B1, B2, R coordinates';
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            initialize(this);
        end
        
        function initialize(this)
            [this.BC_stack0, this.Coord0, ~, ~] = read_grid(this.boundaryFile, 0);
            
            this.Xmax = max(this.Coord0(:,1));
            this.Ymax = max(this.Coord0(:,2));
            this.Xmin = min(this.Coord0(:,1));
            this.Ymin = min(this.Coord0(:,2));
            this.RANGE = [this.Xmin, this.Xmax, this.Ymin, this.Ymax];
        end
        
        function initialState = reset(this)                        
            this.BC_stack = this.BC_stack0;
            this.Coord = this.Coord0;
            
            this.nFaces = size(this.BC_stack,1);
            
            PLOT(this.BC_stack, this.Coord);
            
            BC_stack_sorted = Sort_AFT(this.BC_stack);
            
            node1_base = BC_stack_sorted(1,1);
            node2_base = BC_stack_sorted(1,2);
            PL = NeighborNodes(node1_base, this.BC_stack, node2_base);
            PR = NeighborNodes(node2_base, this.BC_stack, node1_base);
            
            %初始状态为模板点坐标，维度4x2
            initialState = [this.Coord(PL,:);this.Coord(node1_base,:);this.Coord(node2_base,:);this.Coord(PR,:)];
            
            %同时将模板点的标号存起来，后面step时要用
            this.StateIndex = [PL;node1_base;node2_base;PR];
        end
        
        function [observation,reward,isdone,loggedSignals] = step(this,action)
            loggedSignals = [];
            isdone = false;
            
            PL = this.StateIndex(1);
            node1_base = this.StateIndex(2);
            node2_base = this.StateIndex(3);
            PR = this.StateIndex(4);
            base_length = DISTANCE(node1_base, node2_base, this.Coord(:,1), this.Coord(:,2));
            
            %% 当前观察到的状态
            observation =[this.Coord(PL,:);this.Coord(node1_base,:);this.Coord(node2_base,:);this.Coord(PR,:)];
            loggedSignals.State = observation;
            
            %% 当前状态对应的动作
            Pbest = action';% action就是输出的新点坐标
            plot(Pbest(1),Pbest(2),'bo');
            
            %% 计算动作对环境造成的改变，并评估给出奖励
            % 如果Pbest在区域外，则终止
            if (OutOfDomain(Pbest, this.RANGE))
                isdone = true;
                this.IsDone = isdone;
                reward = this.PenaltyForOutOfDomain;
                return;
            end
            
            % 如果Pbest离base front很远，则终止
            Vec = Pbest - this.Coord(node1_base,:);
            dist = sqrt(Vec*Vec');
            if dist > 3 * base_length
                isdone = true;
                this.IsDone = isdone;
                reward = this.PenaltyForOutOfDomain;
                return;
            end
            
            %% 如果Pbest在区域内部
            npoints = size(this.Coord, 1);
            node_select = npoints + 1;
            flag_best = 1;
            for i = 1:npoints
                Vec = this.Coord(i,:) - Pbest;
                dist = sqrt(Vec*Vec');
                if dist < base_length * 0.1     %如果输出的新点离现有点非常近，则选择现有点
                    Pbest = this.Coord(i,:);
                    node_select = i;
                    flag_best = 0;
                    break;
                end
            end
            
            if flag_best == 1   %如果选择了最佳点，则将最佳点坐标加入坐标点序列
                this.Coord(end+1,:) = Pbest;
            end
            %% 考虑周围的点进行相交性判断，若相交，则终止
            nodeCandidate = NodeCandidate(this.BC_stack, node1_base, node2_base, this.Coord(:,1), this.Coord(:,2), Pbest, 3 * base_length);
            frontCandidate = FrontCandidate(this.BC_stack, nodeCandidate);
            flag_not_cross = IsNotCross(node1_base, node2_base, node_select, frontCandidate, this.BC_stack, this.Coord(:,1), this.Coord(:,2), 0);
            if flag_not_cross == 0
                isdone = true;
                this.IsDone = isdone;
                reward = this.PenaltyForCross;
                return;
            end
            
            %% 若不相交，则更新数据，计算奖励
            [this.BC_stack, this.nCells] = UpdateTriCells(this.BC_stack, this.nCells, this.Coord(:,1), this.Coord(:,2), node_select, flag_best);
            
            num_of_new_fronts = size(this.BC_stack,1) - this.nFaces;
            PLOT_NEW_FRONT(this.BC_stack, this.Coord(:,1), this.Coord(:,2), num_of_new_fronts, flag_best)
            
            [this.BC_stack, this.Grid_stack] = DeleteInactiveFront(this.BC_stack, this.Grid_stack);           
            this.nFaces = this.nFaces + num_of_new_fronts;
            this.nNodes = size(this.Coord,1);
            
            quality = TriangleQuality(node1_base, node2_base, node_select, this.Coord(:,1), this.Coord(:,2));
            reward = 10*power(quality, 4);
            
            %% 计算下一步的状态
            BC_stack_sorted = Sort_AFT(this.BC_stack);
            node1_base = BC_stack_sorted(1,1);
            node2_base = BC_stack_sorted(1,2);
            PL = NeighborNodes(node1_base, this.BC_stack, node2_base);
            PR = NeighborNodes(node2_base, this.BC_stack, node1_base);
            observation =[this.Coord(PL,:);this.Coord(node1_base,:);this.Coord(node2_base,:);this.Coord(PR,:)];
            
            this.StateIndex = [PL; node1_base; node2_base; PR];
            loggedSignals.State = observation;           
            this.IsDone = false;

            %% 考虑加上剩余面积判断，填满计算域则给奖励1
            %             if this.nCells == 5
            %                 reward = ;
            %             end
        end
    end
end