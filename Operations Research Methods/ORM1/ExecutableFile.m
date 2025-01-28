clc
clear


load 'planar100.mat'
k = numCom; n = numNodes;
% % comVec(1,3) = 20;

% capMat = [0,1,0,1;1,0,1,0;1,1,0,1;1,1,1,0];
% costMat = capMat;
% comVec = [1,3,1;3,2,2;4,2,1];
% n = 4; k = 3;

%%

Case = 1; showProgress = 1;

tic;
[FMCR, optvalMCR, masterUtopMCR, yMCR, IMCR] = DWLMCF_modified2(capMat, costMat, comVec, n, k, Case, showProgress);
runningtimeMCR = toc;

fprintf('%s is solved in %f seconds.\n    ------------------------    \n', instName, runningtimeMCR);
