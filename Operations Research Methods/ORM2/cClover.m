function [ind] = cClover(x,y)
%[ind] = cClover( x,y )
%Container function for the Clover. 
%For vectors x and y of size n > 0, ind = cClover(x,y) returns ind a logical 
% vector of size n such that ind(i) = 1 (true) if and only if (x(i),y(i))
% in the clover

s = 2.2;
ind = cCircle((s+.5)*(x-.6)+.6,(s+.5)*(y-.5)+.5) | cCircle(s*x, s*(y-1/2)+1/2) | cCircle(1-s*(1-x),s*y) | cCircle(1-s*(1-x),1-s*(1-y));

end

