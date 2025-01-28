function [ind] = cSquare(x,y)
%[ind] = cSquare( x,y )
%Container function for the square. 
%For vectors x and y of size n > 0, ind = cSquare(x,y) returns ind a logical 
% vector of size n such that ind(i) = 1 (true) if and only if (x(i),y(i)) in [0,1]^2  

ind = 0 <= x & x <= 1 & 0 <= y & y <= 1;

end

