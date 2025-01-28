% EXECUTABLE FILE
% clc
clear 
% close all

s = 2.2;
cfunSq = @(x,y) 0 <= x & x <= 1 & 0 <= y & y <= 1;
cfunCi = @(x,y) (2*x-1).^2 + (2*y-1).^2 <= 1;
cfunT = @(x,y) (x+y) >= 0.6 & y <= 0.4*x + 0.6 & x <= 0.4*y + 0.6;
% cfunSt = @(x,y) cTriangle(x,y) | cTriangle(1-x,1-y);
cfunSt2 = @(x,y) cfunT(x,y) | cfunT(1-x,1-y);
% cfunCl = @(x,y) cCircle((s+.5)*(x-.6)+.6,(s+.5)*(y-.5)+.5) | cCircle(s*x, s*(y-1/2)+1/2) | cCircle(1-s*(1-x),s*y) | cCircle(1-s*(1-x),1-s*(1-y));
cfunCl2 = @(x,y) cfunCi((s+.5)*(x-.6)+.6,(s+.5)*(y-.5)+.5) | cfunCi(s*x, s*(y-1/2)+1/2) | cfunCi(1-s*(1-x),s*y) | cfunCi(1-s*(1-x),1-s*(1-y));
containers = {cfunSq, cfunCi, cfunT, cfunSt2, cfunCl2};

n = 50;
cfun = containers{1};
T = 20;
sP = 1;

[d, x, y] = PPP(n, cfun, T, sP);