function drawContainer(Hcfun)
%drawContainer(Hcfun) plots the container corresponding to the function 
% with handle Hcfun.

n=1000;
[X,Y] = meshgrid((0:n)/n,(0:n)/n);
Z = Hcfun(X,Y);
pcolor(X,Y,double(Z))
shading flat
axis equal tight 

end