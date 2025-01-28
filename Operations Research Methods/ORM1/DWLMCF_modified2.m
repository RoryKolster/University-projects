function [F, optval, U, y, I] = DWLMCF_modified2(capMat, costMat, comVec, n, k, Case, showProgress)
% DWLMCF -- solve LMCF problem with Dantzig-Wolfe or branching
% [F, optval, U, y, I] = DWLMCF(capMat, costMat, comVec, n, k, Case, showProgress)

    % For us it is easier to work with transposed matrices mentally for
    % linear indices
    capacityMatrix = capMat';
    costMatrix = costMat';
    
    % Find indices of all existing arcs and amount of existing arcs
    existingArcs = find(capacityMatrix);
    E = length(existingArcs);
    
    % Create long vectors for cap and cost of existing arcs
    capacities = capacityMatrix(existingArcs);
    costs = costMatrix(existingArcs);
    
    % Find initial feasible solution for master or return infeasibility
    [masterConstraint, feasibility] = feasibleInitialSolution(capacities, costs, existingArcs, comVec, n, k);
    if feasibility == 0
        error('The problem instance is infeasible.\n');
    end

    if showProgress == 1
        fprintf('Initial feasible solution for master is found.\n');
    end

    % Initialize a master problem (things that do not change)
    master.A = masterConstraint;
    master.rhs = [capacities; ones(k,1)];
    master.sense = [repmat('<',  E, 1); repmat('=', k, 1)];
    master.modelsense = 'min';
    params.outputflag = 0;

    % Create arrays to store subproblem solutions
    reducedCosts = zeros(1, k);
    optimalSolution = zeros(E, k);
    
    % Create the sub problem (things that do not change)
    subConstraint = zeros(n, E);
    source = ceil(existingArcs/n)';
    SubConstraintInd = 1:E;
    sink = mod(existingArcs, n)';
    sink(sink==0) = n;
    outgoingSubConstraint = sub2ind([n,E], source, SubConstraintInd);
    incomingSubConstraint = sub2ind([n,E], sink, SubConstraintInd);
    subConstraint(outgoingSubConstraint) = -1;
    subConstraint(incomingSubConstraint) = 1;
    sub.A = sparse(subConstraint);
    
    subRHS = zeros(n,k);
    for i=1:k
        subRHS(comVec(i,1), i) = -comVec(i,3);
        subRHS(comVec(i,2), i) = comVec(i,3);
    end
    
    sub.sense = repmat('=', n, 1);
    sub.modelsense = 'min';
    sub.lb = zeros(E, 1);
    sub.vtype = repmat('C', E, 1);

    % setting up the subFlags for subproblems to solve
    subFlag = 1;
    % break condition: if 0 then continue solving, if 1 then break from
    % entire loop (END)
    breakcondition = 0;

    % where to split the subproblems in half
    kLQuart = floor(k/4);
    kHalf = floor(k/2);
    kUQuart = floor(3*k/4);

    % If in Case 2 initialize last_optval
    if Case == 2
        last_optval = Inf;
    end

    % Flow shortest cost-path violating capacity
    totalflow = zeros(1,E);
    for i=1:k
        G = digraph(costMat);
        path = shortestpath(G, comVec(i,1), comVec(i,2));
    
        % Check which arcs are activated from path
        sentArcs = path(1:end-1);
        arriveArcs = path(2:end);
        activatedArcs = sub2ind([n, n], arriveArcs, sentArcs);
        usedArcInd = ismember(existingArcs, activatedArcs);
        totalflow(usedArcInd)= totalflow(usedArcInd) + comVec(i,3);
    end

    % Lower bound violating capacity
    lbViolatedCap = totalflow * costs;
    
    loopCounter = 0;
    while 1
        loopCounter = loopCounter + 1;

        % Solve current master problem
        master.obj = costs' * masterConstraint(1:E,:);
        master.A = masterConstraint;
        master.lb = zeros(size(masterConstraint, 2), 1);
        master.vtype = repmat('C', size(masterConstraint, 2), 1);
        masterResults = gurobi(master, params);
        optval = masterResults.objval;
        shadowsMaster = masterResults.pi;

        % Calculate beta bound for problem (only needs to be done once as
        % the amount of flow over complete problem does not change)
        if loopCounter == 1
            beta = sum(masterConstraint(1:E,:) * masterResults.x) + 1;
        end
        
        % If Case 2 is prompted implement stopping criteria based on last
        % two objective values of the master problem
        if Case == 2
            if (last_optval - optval)/optval <  10^-4 && last_optval ~= optval
                fprintf('Percentage difference between consequent objective values is less than 10^-4 so stopping criteria is activated. \n    ------------------------    \n')
                fprintf('Loop %i   -   obj. master %f \n    ------------------------    \n', loopCounter, optval)
                break
            end
        end

        % Set objective for sub problems in current iteration
        sub.obj = costs - shadowsMaster(1:E);

        % Sub while loop
        secondFlag = 0;
        while 1
        
            % Set beginning and end of subproblems for each subFlag case
            if secondFlag == 0
                if subFlag == 1
                    beginSubs = 1;
                    endSubs = kLQuart;
                elseif subFlag == 2
                    beginSubs = kLQuart + 1;
                    endSubs = kHalf - 1;
                elseif subFlag == 3
                    beginSubs = kHalf + 1;
                    endSubs = kUQuart - 1;
                elseif subFlag == 4
                    beginSubs = kUQuart + 1;
                    endSubs = k;
                % else
                %     beginSubs = 1;
                %     endSubs = k;
                end
            else
                if subFlag == 1 || subFlag == 2
                    beginSubs = 1;
                    endSubs = kHalf;
                elseif subFlag == 3 || subFlag == 4
                    beginSubs = KHalf + 1;
                    endSubs = k;
                else
                    beginSubs = 1;
                    endSubs = k;
                end
            end
                 
            % For corresponding subproblems calculate reduced costs and store
            % optimal x vector
            for i=beginSubs:endSubs
                sub.rhs = subRHS(:, i);
                subResults = gurobi(sub, params);
                reducedCosts(i) = subResults.objval - shadowsMaster(E + i);
                optimalSolution(:, i) = subResults.x;
            end
        
            % Check which problems can result in lower opt value of master
            indexRC = find(reducedCosts(beginSubs:endSubs) < -0.00001);
        
            % If empty and first time, go to other have
            if isempty(indexRC)
                % First quarter empty, check first half
                if secondFlag == 0 & subFlag == 1
                    secondFlag = 1;
                    subFlag = 2;
                    continue;
        
                % Second quarter empty, check second half
                elseif secondFlag == 0 & subFlag == 2
                    secondFlag = 1;
                    subFlag = 3;
                    continue;
                
                % Third quarter empty, check second half
                elseif secondFlag == 0 & subFlag == 3
                    secondFlag = 1;
                    subFlag = 4;
                    continue;
                   
                % Fourth quarter empty, check first half
                elseif secondFlag == 0 & subFlag == 4
                    secondFlag = 1;
                    subFlag = 1;
                    continue;

                % First half empty, check second half
                elseif secondFlag == 1 & (subFlag == 1 || subFlag == 2)
                    secondFlag = 3;
                    subFlag = 3;
                    continue;
                  
                % Second half empty, check first half
                elseif secondFlag == 1 & (subFlag == 3 || subFlag == 4)
                    secondFlag = 3;
                    subFlag = 1;
                    continue;
                 
        
                % Both halfs are checked and thus empty, no neg. reduced costs
                else
                    breakcondition = 1;
                    break
                end
            end
        
            % If both subsets needed to be used, from now on only use all sub
            % problems
            if secondFlag == 1
                subFlag = 5;
                break
            end
        
            % If less than 10% of subproblems was helpful, go to full sub problems
            % Otherwise, go to other half in next rotation
            if secondFlag == 0 || secondFlag == 1
                percentage = (nnz(indexRC))/(k/2);
        
                % If #red cost < 0 < 10%, go to full sub problems and break loop
                % for now
                if percentage < 0.1
                    subFlag = 5;
                    break
                
                % Else go to other quarter/half next rotation and break loop
                else
                    if secondFlag == 0
                        if subFlag == 1
                            subFlag = 2;
                        elseif subFlag == 2
                            subFlag = 3;
                        elseif subFlag == 3
                            subFlag = 4;
                        elseif subFlag == 4
                            subFlag = 1;
                        end
                    elseif secondFlag == 1
                        if subFlag == 1 || subFlag == 2
                            subFlag = 3;
                        elseif subFlag == 3 || subFlag == 4
                            subFlag = 1;
                        end
                    end
                    break
                end
            end
        end        
        
        % Show progress if necessary
        if showProgress == 1
            [bestMPS, indexMPS] = min(reducedCosts, [], "all");
            fprintf('Loop %i   -   obj. master %f   -   obj. most promising sub %f   -   index of most promising sub %i   -   lb. %f   -   ub. %f\n', loopCounter, optval, bestMPS, indexMPS, max(optval + bestMPS * beta, lbViolatedCap), optval);
        end
        
        %Break condition
        if breakcondition == 1
            break
        end
        
        % Create new constraints for lambdas relative to indexRC and the starting
        % position of subproblems
        newMasterConstraint = zeros(k, length(indexRC));
        mastersize2 = size(masterConstraint, 2);
        newMasterConstraint(sub2ind([k, mastersize2], indexRC + beginSubs - 1, 1:length(indexRC))) = 1;
        
        % Add new patterns relative to starting position of subproblems
        newPattern = [optimalSolution(:, indexRC + beginSubs - 1); newMasterConstraint];
        masterConstraint = sparse([masterConstraint, newPattern]);
        
        % If in Case 2 set last objective of master to last_optval
        if Case == 2
            last_optval = optval;
        end
        
        
    end

    % Return what is asked
    y = masterResults.x;

    F = zeros(n,n);
    F(existingArcs) = masterConstraint(1:E,:) * y;
    F = F';

    U = masterConstraint;

    [~, I] = max(masterConstraint(E+1:end,:), [], 1);

    % fprintf('It took %i loops   -   obj. %f \n', loopCounter, optval);
end

function [U, feasible] = feasibleInitialSolution(capacities, costs, existingArcs, comVec, n, k)
% feasibleInitialSolution -- finds initial feasible solution OR returns infeasibility
% 
% [U, feasible] = feasibleInitialSolution(capacities, costs, existingArcs, comVec, n, k)

    % Find initial feasible solution for master problem
    E = length(existingArcs);
    copyCapacities = capacities;

    % Initialize a master problem (things that do not change), we need this
    % to check if problem is feasible
    master.rhs = [capacities; ones(k,1)];
    master.sense = [repmat('<',  E, 1); repmat('=', k, 1)];
    master.modelsense = 'min';
    params.outputflag = 0;

    % First try a naive way to formulate problem, if that results in a
    % feasible solution, do not use exhaustive search
    [masterConstraint, feasibilityNaive] = LocalNaiveSolution(capacities, existingArcs, comVec, n, k);
    if feasibilityNaive == 1
        master.A = masterConstraint;
        master.obj = zeros(size(masterConstraint, 2), 1);
        master.lb = zeros(size(masterConstraint, 2), 1);
        master.vtype = repmat('C', size(masterConstraint, 2), 1);
        masterResults = gurobi(master, params);
    
        % Check if the master problem is feasible
        if strcmp(masterResults.status, 'OPTIMAL')
            U = master.A;
            feasible = 1;
            return      
        end
    end

    % Create source and sink array of arcs
    Originalsource = ceil(existingArcs/n);
    Originalsink = mod(existingArcs, n);
    Originalsink(Originalsink==0) = n;
    
    % Store existing paths
    existingPaths = cell(k, 1);
    restrictionArcs = cell(k, 1);
    capacityMet = zeros(k,1);
    exhaustSearched = cell(k,1);
    
    % Initialize arrays for index-value pairs of sparse matrix
    rowIndices = [];
    columnIndices = [];
    valueIndices = [];
    rowLambdas = [];
    columnLambdas = [];

    % For loop for finding absolute shortest path for each commodity
    for i=1:k

        % source, sink and demand for current commodity
        origin = comVec(i,1);
        destination = comVec(i,2);
        demand = comVec(i,3);

        % Create a graph, find the shortest path and add it to all existing
        % paths. No arcs are not allowed, so restrictions is empty
        G = digraph(Originalsource, Originalsink);
        path = shortestpath(G, origin, destination);
        existingPaths{i} = {path};
        restrictionArcs{i} = {[]};

        % The path has not been exhaustively searched for new paths
        exhaustSearched{i} = 0;
    
        % Check path length
        pathLength = length(path);
    
        % Check which arcs are activated from path
        sentArcs = path(1:end-1);
        arriveArcs = path(2:end);
        activatedArcs = sub2ind([n, n], arriveArcs, sentArcs);
        usedArcInd = ismember(existingArcs, activatedArcs);
    
        % What capacity can this path hold
        capPath = copyCapacities(usedArcInd);
        capacityMet(i) = capacityMet(i) + min(capPath);
    
        % Add the path to the sparse matrix
        corRowInd = find(usedArcInd)';
        rowIndices = [rowIndices, corRowInd];
        columnIndices = [columnIndices, i * ones(1, pathLength-1)];
        valueIndices = [valueIndices, demand * ones(1, pathLength-1)];

        % Add the indices for corresponding lambda 
        rowLambdas = [rowLambdas, i];
        columnLambdas = [columnLambdas, i];
    end

    % If length of row and col lambda is not k, instant infeasible. there
    % is no path for at least one commodity, without any restrictions
    rowLength = length(rowLambdas);
    columnLength = length(columnLambdas);
    if rowLength ~= k
        U = 0;
        feasible = 0;
        return 
    end
    if columnLength ~= k
        U = 0;
        feasible = 0;
        return 
    end
    
    % flag for feasibility of master and amount of paths found
    flagMaster = 0;
    pathCounter = k+1;    
    
    % While master is infeasible, add best not used shortest path for each
    % commodity
    while flagMaster == 0

        % Amount of new paths in current iteration for all commodities
        amountNP = 0;
    
        % For loop for adding new not used path for each commodity
        for i=1:k
            
            % source, sink and demand for current commodity
            origin = comVec(i,1);
            destination = comVec(i,2);
            demand = comVec(i,3);

            % already existing paths for current commodity and for each
            % path what restrictions were used
            currentPaths = existingPaths{i};
            currentRestrictions = restrictionArcs{i};
            exhaustedPaths = exhaustSearched{i};

            % amount of paths existing
            amountCP = length(currentPaths);

            if all(exhaustedPaths == 1)
                continue
            end
            
            % for each existing path, check if a new path can be
            % found
            for j=1:amountCP

                % If the path has been searched exhaustively, then no use
                % in checking again
                if exhaustedPaths(j) == 1
                    continue
                end

                % Take the path from which we are going to deviate, check
                % its length and what restrictions it has
                oldPath = currentPaths{j};
                amountoP = length(oldPath)-1;
                pathRestrictions = currentRestrictions{j};

                % Flag if a new path is found
                flagPath = 0;
        
                % for each arc in old path, check if there is a new path
                % possible if that arc is removed
                for l=1:amountoP

                    % Find index of the arc being removed
                    [indX,~] = find(existingArcs == sub2ind([n,n], oldPath(l+1), oldPath(l)));

                    % All available arcs, so also not earlier restrictions
                    availableArcs = find(~ismember(1:E,[pathRestrictions, indX]));

                    % Create new graph and find new path
                    source = Originalsource(availableArcs);
                    sink = Originalsink(availableArcs);
                    Graph = digraph(source, sink);
                    newPath = shortestpath(Graph, origin, destination);

                    % If the path is empty, there is no path possible, go
                    % to next arc
                    if isempty(newPath)
                        continue
                    end
                    
                    % If new path not in existing paths (due to bad coding
                    % some paths will be regenerated) add it
                    if ~any(cellfun(@(x) isequal(x, newPath), currentPaths))
                        % Add new path to existing paths
                        currentPaths = [currentPaths; newPath];
                        existingPaths{i} = currentPaths;
                        
                        % Add restriction by finding new path
                        currentRestrictions = [currentRestrictions, [pathRestrictions, indX]];
                        restrictionArcs{i} = currentRestrictions;

                        % New path is not exhaustively searched
                        exhaustedPaths = [exhaustedPaths 0];
                        exhaustSearched{i} = exhaustedPaths;

                        % New path is found
                        flagPath = 1;
        
                        % Check path length
                        pathLength = length(newPath);
        
                        % Check which arcs are activated from path
                        sentArcs = newPath(1:end-1);
                        arriveArcs = newPath(2:end);
                        activatedArcs = sub2ind([n, n], arriveArcs, sentArcs);
                        usedArcInd = ismember(existingArcs, activatedArcs);
    
                        % What capacity can this path hold
                        capPath = copyCapacities(usedArcInd);
                        capacityMet(i) = capacityMet(i) + min(capPath);
    
                        % Add the path to the sparse matrix
                        corRowInd = find(usedArcInd)';
                        rowIndices = [rowIndices, corRowInd];
                        columnIndices = [columnIndices, pathCounter * ones(1, pathLength-1)];
                        valueIndices = [valueIndices, demand * ones(1, pathLength-1)];

                        % Add row and col lambda
                        rowLambdas = [rowLambdas, i];
                        columnLambdas = [columnLambdas, pathCounter];

                        % Amount of total paths (patterns)
                        pathCounter = pathCounter + 1;

                        % Update amount of paths found this complete while loop
                        amountNP = amountNP + 1;
                        break
                    end
                end

                % If a new path is found, go to next commodity
                if flagPath == 1
                    break
                end

                % If no new path is found, the path has been searched
                % exhaustively
                if flagPath == 0
                    exhaustedPaths(j) = 1;
                    exhaustSearched{i} = exhaustedPaths;
                end
            end
        end
    
        % If no new paths are added, the problem is infeasible, as it was
        % infeasible in the previous iteration
        if amountNP == 0
            U = masterConstraint;
            feasible = 0;
            return
        end
    
        % If total capacity for commodity is below the demand, never feasible
        if any(capacityMet < comVec(i,3))
            continue
        end
        
        % Create sparse matrix
        masterUtop = sparse(rowIndices, columnIndices, valueIndices, E, pathCounter-1);
        
        % Create lambda==1 constraint
        masterUbottom = sparse(rowLambdas, columnLambdas, ones(1, pathCounter-1), k, pathCounter-1);
    
        % Solve current master problem
        masterConstraint = sparse([masterUtop; masterUbottom]);
        master.obj = costs' * masterUtop;
        master.A = masterConstraint;
        master.lb = zeros(size(masterConstraint, 2), 1);
        master.vtype = repmat('C', size(masterConstraint, 2), 1);
        masterResults = gurobi(master, params);

        % Check if the master problem is feasible
        if strcmp(masterResults.status, 'OPTIMAL')
            U = masterConstraint;
            feasible = 1;
            return      
        end
        % fprintf('Currently there are %i patterns.\n', pathCounter - 1);
    end
end

function [U, feasible] = LocalNaiveSolution(capacities, existingArcs, comVec, n, k)
% LocalNaiveSolution -- finds naive initial solution without exhaustive
% search
%
% U = LocalNaiveSolution(capacities, costs, existingArcs, comVec, n, k)

    E = length(existingArcs);
    copyCapacities = capacities;

    % Initialize arrays for index-value pairs of sparse matrix
    rowIndices = [];
    columnIndices = [];
    valueIndices = [];

    % Initialize feasible to be true
    feasible = 1;
    
    i = 1;
    demandMet = 0;
    while i <= k
        % Find shortest path, not considering costs
        remainingArcInd = copyCapacities > 0;
        remainingArcs = existingArcs(remainingArcInd);
        source = ceil(remainingArcs/n);
        sink = mod(remainingArcs, n);
        sink(sink==0) = n;
        G = digraph(source, sink);
        path = shortestpath(G, comVec(i,1), comVec(i,2));
    
        % If the path is empty, there is no path and go to next commodity
        if isempty(path)
            if demandMet < comVec(i,3)
                feasible = 0;
                break
            end
            i = i + 1;
            continue
        end
    
        % Check path length 
        pathLength = length(path);
    
        % Check which arcs are activated from path
        sentArcs = path(1:end-1);
        arriveArcs = path(2:end);
        activatedArcs = sub2ind([n, n], arriveArcs, sentArcs);
    
        % Check what the minimum capacity and thus binding arc is
        % if the minimum flowsize is larger than demand, only use the demand
        usedArcInd = ismember(existingArcs, activatedArcs);
        capPath = copyCapacities(usedArcInd)';
        flowsize = min([capPath, comVec(i,3) - demandMet]);
    
        % Reduce capacity on used arcs with flowsize
        copyCapacities(usedArcInd) = copyCapacities(usedArcInd) - flowsize;
    
        % Create the pattern via index-value pairs
        corRowInd = find(usedArcInd)';
        rowIndices = [rowIndices, corRowInd];
        columnIndices = [columnIndices, i * ones(1, pathLength-1)];
        valueIndices = [valueIndices, flowsize * ones(1, pathLength-1)];
    
        % If the demand is met, go to the next commodity and set demandMet to 0
        demandMet = demandMet + flowsize;
        if demandMet >= comVec(i,3)
            demandMet = 0;
            i = i + 1;
        end
    end


    % If naive path finding is infeasible due to no available paths, return
    % this
    if feasible == 0
        U = 0;
        return
    end

    % Create sparse matrix
    masterUtop = sparse(rowIndices, columnIndices, valueIndices, E, k);
    
    % Create lambda==1 constraint
    masterUbottom = speye(k);

    U = sparse([masterUtop; masterUbottom]);
end