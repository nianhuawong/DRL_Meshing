classdef (Abstract) mesh_DRL_Abstract < rl.env.MATLABEnvironment
    properties
        nNodes     = 0;
        nFaces     = 0;
        nCells     = 0;
        Grid_stack = [];
        boundaryFile =  './boundary_file.cas';
        Coord = [];
    end
    
    properties
        % system state [L, B1, B2, R]'
        State = zeros(4,1)
    end
    
    properties(Access = protected)
        % Internal flag to store stale env that is finished
        IsDone = false
    end
    
    methods
        function this = mesh_DRL_Abstract(ActionInfo)
            %% 状态为输入模板点
            ObservationInfo = rlNumericSpec([4 1]);
            ObservationInfo.Name = 'mesh DRL States';
            ObservationInfo.Description = 'L, B1, B2, R';
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
        end
        
        function initialState = reset(this)
            
            [BC_stack, this.Coord, ~, ~] = read_grid(this.boundaryFile, 0);
            PLOT(BC_stack, this.Coord);
            
            BC_stack_sorted = Sort_AFT(BC_stack);
            
            base1 = BC_stack_sorted(1,1);
            base2 = BC_stack_sorted(1,2);
            PL = NeighborNodes(base1, BC_stack, base2);
            PR = NeighborNodes(base2, BC_stack, base1);
            
            initialState = [PL;base1;base2;PR];%只存点的编号
            this.State = initialState;
        end
        
        function [observation,reward,isdone,loggedSignals] = step(this,action)
            loggedSignals = [];
            % action就是输出的新点坐标
            newpoint = action;
            
            %             PL = this.State(1);
            node1_base = this.State(2);
            node2_base = this.State(3);
            %             PR = this.State(4);
            base_length = DISTANCE(node1_base, node2_base, this.Coord(:,1), this.Coord(:,2));
            
            npoints = size(this.Coord, 1);
            node_select = npoints + 1;
            flag_best = 1;
            for i = 1:npoints
                VV = this.Coord(i,:) - newpoint;
                dist = sqrt(VV.*VV);
                if dist < base_length * 0.1     %如果输出的新点离现有点非常近，则选择现有点
                    newpoint = this.Coord(i,:);
                    node_select = i;
                    flag_best = 0;
                    break;
                end
            end
            
            if flag_best == 1   %如果选择了最佳点，则将最佳点坐标加入坐标点序列
                this.Coord(end+1,:) = newpoint;
            end
            %%
            [this.BC_stack, this.nCells] = UpdateTriCells(this.BC_stack, this.nCells, this.Coord(:,1), this.Coord(:,2), node_select, flag_best);
            
            [this.BC_stack, this.Grid_stack] = DeleteInactiveFront(this.BC_stack, this.Grid_stack);
            
            %%
            BC_stack_sorted = Sort_AFT(this.BC_stack);
            
            node1_base = BC_stack_sorted(1,1);
            node2_base = BC_stack_sorted(1,2);
            PL = NeighborNodes(node1_base, this.BC_stack, node2_base);
            PR = NeighborNodes(node2_base, this.BC_stack, node1_base);
            observation = [PL, node1_base, node2_base, PR];
            
            loggedSignals.State = observation;
            this.State = observation;
                        
            %%
            isdone = false;
            nodeCandidate = NodeCandidate(this.BC_stack, node1_base, node2_base, this.Coord(:,1), this.Coord(:,2), newpoint, 3 * base_length);
            frontCandidate = FrontCandidate(this.BC_stack, nodeCandidate);
            flag_not_cross = IsNotCross(node1_base, node2_base, newpoint, frontCandidate, this.BC_stack, this.Coord(:,1), this.Coord(:,2), 0);
            if flag_not_cross == 0
                isdone = true;
            end
            this.IsDone = isdone;
            %%
            quality = TriangleQuality(node1_base, node2_base, node_select, this.Coord(:,1), this.Coord(:,2));
            if this.IsDone
                reward = -10;
            else
                reward = quality;
            end
            
            %% 考虑加上剩余面积判断，填满计算域则给奖励1
            
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
    end
end