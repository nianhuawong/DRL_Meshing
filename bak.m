            %%
            [x_new, y_new] = ADD_POINT_tri(this.BC_stack, this.Coord(:,1), this.Coord(:,2), 0, 1.0);
            action =  action./norm(action);   
            Pbest = [x_new, y_new] + action' * base_length;
%%
%             [x_new, y_new] = ADD_POINT_tri(this.BC_stack, this.Coord(:,1), this.Coord(:,2), 0, 1.0);
%             action(1:2) =  action(1:2)./norm(action(1:2));            
%             al = 1.0 * base_length;
%             Pbest = [x_new, y_new] + al * action(1:2)' * abs(action(3));