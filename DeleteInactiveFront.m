function [AFT_stack, Grid_stack, num_deleted_fronts] = DeleteInactiveFront(AFT_stack, Grid_stack)
num_deleted_fronts = 0;
for i = 1: size(AFT_stack,1)
    if((AFT_stack(i,3) ~= -1) && (AFT_stack(i,4) ~= -1))  %左单元和右单元编号均不为-1
        Grid_stack(end+1,:) = AFT_stack(i,:);
        %             iFace = iFace + 1;
        AFT_stack(i,:)=-1;
        num_deleted_fronts = num_deleted_fronts + 1;
    end
end

AFT_stack( AFT_stack(:,1) == -1, : ) = [];

end