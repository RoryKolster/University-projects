function [ind] = cStar(x,y)
%[ind] = cStar( x,y )
%Container function for the Star. 
%For vectors x and y of size n > 0, ind = cStar(x,y) returns ind a logical 
% vector of size n such that ind(i) = 1 (true) if and only if (x(i),y(i))
% in the star 

ind = cTriangle(x,y) | cTriangle(1-x,1-y);

end

