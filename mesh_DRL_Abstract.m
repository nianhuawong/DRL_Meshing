classdef (Abstract) mesh_DRL_Abstract < rl.env.MATLABEnvironment
    properties
        nNodes     = 0;
        nFaces     = 0;
        nCells     = 0;
        nFronts    = 0;
        Grid_stack = [];
        BC_stack0  = [];
        BC_stack   = [];
        boundaryFile =  './grid/boundary_file.cas';
        Coord0 = [];
        Coord  = [];
        
        RANGE = [];
        
        PenaltyForOutOfDomain = -10;
        PenaltyForCross = -10;

        isPlot = 1;
        standardlize = 1;
    end
    
    properties
        StateIndex = zeros(4,1)
    end
    
    properties(Access = protected)
        IsDone = false
        lastobs
    end
    
    methods
        function this = mesh_DRL_Abstract(ActionInfo)
            ObservationInfo(1) = rlNumericSpec([656 875 3]);
            ObservationInfo(1).Name = 'image';
            
            ObservationInfo(2) = rlNumericSpec([4 2]);
            ObservationInfo(2).Name = 'state';
            
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            initialize(this);
        end
        
        function initialize(this)
            [this.BC_stack0, this.Coord0, ~, ~] = read_grid(this.boundaryFile, 0);
            
            this.RANGE = [min(this.Coord0(:,1)), max(this.Coord0(:,1)), min(this.Coord0(:,2)), max(this.Coord0(:,2))];         
        end
        
        function ObsInfo = reset(this) 
            this.Grid_stack = [];
            this.BC_stack = this.BC_stack0;
            this.Coord = this.Coord0;
            
            if(this.isPlot)
                PLOT(this.BC_stack, this.Coord);
            end   
            
            this.nCells  = 0;
            this.nNodes  = size(this.Coord,1);
            this.nFaces  = size(this.BC_stack,1);
            this.nFronts = size(this.BC_stack,1);
            
            BC_stack_sorted = Sort_AFT(this.BC_stack);
            
            node1_base = BC_stack_sorted(1,1);
            node2_base = BC_stack_sorted(1,2);
            PL = FindOneNeighborNode(node1_base, this.BC_stack, node2_base);
            PR = FindOneNeighborNode(node2_base, this.BC_stack, node1_base);
            
            %初始状态为模板点坐标 
            state = [this.Coord(PL,:);this.Coord(node1_base,:);this.Coord(node2_base,:);this.Coord(PR,:)];
            
            if this.standardlize
                state = Standardlize(state);
            end
            
            %同时将模板点的标号存起来，后面step时要用
            this.StateIndex = [PL;node1_base;node2_base;PR];
%%
            image = importdata('./env0.png');
            
            ObsInfo = {image, state};
            this.lastobs = ObsInfo;
        end
        
        function [nextobs,reward,isdone,loggedSignals] = step(this,action)
            loggedSignals = [];
            nextobs = this.lastobs;
            isdone = false;
            
            PL = this.StateIndex(1);
            node1_base = this.StateIndex(2);
            node2_base = this.StateIndex(3);
            PR = this.StateIndex(4);
            base_length = DISTANCE(node1_base, node2_base, this.Coord(:,1), this.Coord(:,2));
            
            if(this.isPlot)
                PLOT_FRONT(this.BC_stack, this.Coord(:,1), this.Coord(:,2), 1);
            end
            
            %% 当前状态对应的动作
            Pbest = action';% action就是输出的新点坐标 
            if this.standardlize
                Pbest = AntiStandardlize(Pbest, this.Coord(node1_base,:), this.Coord(node2_base,:));
            end
            
            if(this.isPlot)
                plot(Pbest(1),Pbest(2),'bo');
            end
            
            %% 计算动作对环境造成的改变，并评估给出奖励           
            % 如果Pbest离base front很远，则终止
            Vec = Pbest - this.Coord(node1_base,:);
            dist = sqrt(Vec*Vec');
            if dist > 3 * base_length
                isdone = true;
                this.IsDone = isdone;
                reward = this.PenaltyForOutOfDomain*dist;
                return;
            end
            
            %% 如果Pbest与现有点很靠近，则选择现有点
            PSelect = [];
            flag_best = 1;
            index = PointClose2ExistedPoints(Pbest,this.Coord,node1_base,node2_base);
            if index > 0
                node_select = index;
                PSelect = this.Coord(node_select,:);
                flag_best = 0;
                
                [~, row1] = FrontExist(node1_base,node_select, this.Grid_stack);
                [~, row2] = FrontExist(node2_base,node_select, this.Grid_stack);
                if(row1 > 0 || row2 > 0)
                    isdone = true;
                    this.IsDone = isdone;
                    reward = this.PenaltyForOutOfDomain;
                    return;
                end
            end
            
            %% 如果Pbest在区域外，且不与现有点靠近，则终止
            if (OutOfDomain(Pbest, this.RANGE) && index < 0)
                isdone = true;
                this.IsDone = isdone;
                reward = this.PenaltyForOutOfDomain;
                return;
            end
            
            if flag_best == 1   %如果选择了最佳点，则将最佳点坐标加入坐标点序列
                PSelect = Pbest;
                this.Coord(end+1,:) = Pbest;
                node_select = size(this.Coord, 1);
            end
            
            %% 判断是否为左单元
            flag_left_cell = IsLeftCell(node1_base, node2_base, node_select, this.Coord(:,1), this.Coord(:,2));
            if flag_left_cell == 0
                isdone = true;
                this.IsDone = isdone;
                reward = this.PenaltyForCross;
                return;
            end
            
            %% 考虑周围的点进行相交性判断，若相交，则终止
            nodeCandidate = NodeCandidate(this.BC_stack, node1_base, node2_base, this.Coord(:,1), this.Coord(:,2), PSelect, 3 * base_length);
            frontCandidate = FrontCandidate(this.BC_stack, nodeCandidate);
            flag_not_cross = IsNotCross(node1_base, node2_base, node_select, frontCandidate, this.BC_stack, this.Coord(:,1), this.Coord(:,2), 0);
            if flag_not_cross == 0
                isdone = true;
                this.IsDone = isdone;
                reward = this.PenaltyForCross;
                return;
            end
            
            %% 若不相交，则更新数据           
            [this.BC_stack, this.nCells] = UpdateTriCells(this.BC_stack, this.nCells, this.Coord(:,1), this.Coord(:,2), node_select, flag_best); 
            
            num_of_new_fronts = size(this.BC_stack,1) - this.nFronts;
            if(this.isPlot)
                PLOT_NEW_FRONT(this.BC_stack, this.Coord(:,1), this.Coord(:,2), num_of_new_fronts, flag_best)
            end
            
            [this.BC_stack, this.Grid_stack, num_deleted_fronts] = DeleteInactiveFront(this.BC_stack, this.Grid_stack);  
            this.nFronts = this.nFronts + num_of_new_fronts - num_deleted_fronts; 
            this.nFaces = this.nFaces + num_of_new_fronts;
            this.nNodes = size(this.Coord,1);
            
            %% 计算奖励            
            quality = TriangleQuality(node1_base, node2_base, node_select, this.Coord(:,1), this.Coord(:,2));
            if flag_best == 0
                quality = quality / 0.8;
            end
            
            reward = power(quality, 1);
            
            %% 
            if(isempty(this.BC_stack))
                isdone = true;
                this.IsDone = isdone;
                reward = this.RewardForFinishDomain;
                return;
            end
                
            %% 计算下一步的状态
            BC_stack_sorted = Sort_AFT(this.BC_stack);
            node1_base = BC_stack_sorted(1,1);
            node2_base = BC_stack_sorted(1,2);
            PL = FindOneNeighborNode(node1_base, this.BC_stack, node2_base);
            PR = FindOneNeighborNode(node2_base, this.BC_stack, node1_base);
            state =[this.Coord(PL,:);this.Coord(node1_base,:);this.Coord(node2_base,:);this.Coord(PR,:)];

            if this.standardlize
                state = Standardlize(state);
            end
            
            saveas(gcf,'env.png');
            image=importdata('./env.png');
            
            this.lastobs = nextobs;
            nextobs = {image, state};
             
            this.StateIndex = [PL; node1_base; node2_base; PR];         
            this.IsDone = false;
        end
    end
end