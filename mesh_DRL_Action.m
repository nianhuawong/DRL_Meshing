classdef mesh_DRL_Action < mesh_DRL_Abstract
    
    methods
        function this = mesh_DRL_Action()
            %% 动作为生成一个新点坐标
            ActionInfo = rlNumericSpec([2 1]);
            ActionInfo.Name = 'mesh DRL Action';
            
            this = this@mesh_DRL_Abstract(ActionInfo);
        end
    end
    
end