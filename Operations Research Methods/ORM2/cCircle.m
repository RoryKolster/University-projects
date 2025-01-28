function [ind] = cCircle(x,y)
%[ind] = cCircle( x,y )
%Container function for the Circle. 
%For vectors x and y of size n > 0, ind = cCircle(x,y) returns ind a logical 
% vector of size n such that ind(i) = 1 (true) if and only if 
% (2*x(i)-1)^2 + (2*y(i)-1)^2 <= 1
% 

ind = (2*x-1).^2 + (2*y-1).^2 <= 1;

end

