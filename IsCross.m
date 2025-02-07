function flag = IsCross(a, b, c, d)
global crossCount;
crossCount = crossCount + 1;
    flag = 0;
    
    if( (  min(a(1),b(1)) <= max(c(1),d(1)) ...  % 1.ab的最左端小于cd的最右端
        && min(c(1),d(1)) <= max(a(1),b(1)) ...  % 2.cd的最左端小于ab的最右端
        && min(a(2),b(2)) <= max(c(2),d(2)) ...  % 4.ab的最低点低于cd的最高点
        && min(c(2),d(2)) <= max(a(2),b(2))))    % 3.cd的最低点低于ab的最高点
%%    
%         ab = b - a; ab(3) = 0;
%         ac = c - a; ac(3) = 0;
%         ad = d - a; ad(3) = 0;
% 
%         ca = a - c; ca(3) = 0;
%         cb = b - c; cb(3) = 0;
%         cd = d - c; cd(3) = 0;         
%         tmp1 = cross(ac, ab) * transpose( cross(ad, ab) );    % ac x ab   % ad x ab
%         tmp2 = cross(ca, cd) * transpose( cross(cb, cd) );    % ca x cd   % cb x cd
%%
        u=(c(1)-a(1))*(b(2)-a(2))-(b(1)-a(1))*(c(2)-a(2));
        v=(d(1)-a(1))*(b(2)-a(2))-(b(1)-a(1))*(d(2)-a(2));
        w=(a(1)-c(1))*(d(2)-c(2))-(d(1)-c(1))*(a(2)-c(2));
        z=(b(1)-c(1))*(d(2)-c(2))-(d(1)-c(1))*(b(2)-c(2));
             
        eps = 1e-9;

%         if( tmp1 <= 0.00000001 && tmp2 <= 0.00000001 )                  %叉乘的乘积小于0，应该是相交
        if( u*v <= 0 && w*z <= 0 )    
            if(  (( abs( a(1) - c(1) ) < eps ) && ( abs( a(2) - c(2) ) < eps )) ...
                    ||(( abs( a(1) - d(1) ) < eps ) && ( abs( a(2) - d(2) ) < eps )) ...   %但是，只要有一个点重叠，则认为不相交
                    ||(( abs( b(1) - c(1) ) < eps ) && ( abs( b(2) - c(2) ) < eps )) ...
                    ||(( abs( b(1) - d(1) ) < eps ) && ( abs( b(2) - d(2) ) < eps )) )

                flag = 0;
            else
                flag = 1;
            end
        end
   end
    
%%    
%     if( (  min(a(1),b(1)) < max(c(1),d(1)) ...  % 1.ab的最左端小于cd的最右端
%         && min(c(1),d(1)) < max(a(1),b(1)) ...  % 2.cd的最左端小于ab的最右端
%         && min(a(2),b(2)) < max(c(2),d(2)) ...  % 4.ab的最低点低于cd的最高点
%         && min(c(2),d(2)) < max(a(2),b(2))))    % 3.cd的最低点低于ab的最高点
% 
%         u=(c(1)-a(1))*(b(2)-a(2))-(b(1)-a(1))*(c(2)-a(2));
%         v=(d(1)-a(1))*(b(2)-a(2))-(b(1)-a(1))*(d(2)-a(2));
%         w=(a(1)-c(1))*(d(2)-c(2))-(d(1)-c(1))*(a(2)-c(2));
%         z=(b(1)-c(1))*(d(2)-c(2))-(d(1)-c(1))*(b(2)-c(2));
%         flag = (u*v<=0.00000001 && w*z<=0.00000001);    
%     end

% %             if ( (   a(1) == c(1) && a(2)~= c(2) ) ...
% %                 || ( a(1)~= d(1) && a(2)~= d(2) ) ...
% %                 || ( b(1)~= c(1) && b(2)~= c(2) )...
% %                 || ( b(1)~= d(1) && b(2)~= d(2) ) )
end