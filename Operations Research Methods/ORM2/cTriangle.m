function [ind] = cTriangle(x,y)
%[ind] = cTriangle( x,y )
%Container function for the Triangle. 
%For vectors x and y of size n > 0, ind = cTriangle(x,y) returns ind a logical 
% vector of size n such that ind(i) = 1 (true) if and only if (x(i),y(i))
% in the trangle defined by 
% x+y >= 0.6  
% y <= 0.4*x + 0.6 
% x <= 0.4*y + 0.6 

ind = (x+y) >= 0.6 & y <= 0.4*x + 0.6 & x <= 0.4*y + 0.6 ;

end

