function [d, x, y] = PPP(n, cfun, T, sP)

    % Input parameters
    tabu_len = 10; 
    max_time_no_improv = 0.2 * T;
    point_directions = [1,0;sqrt(2)/2,sqrt(2)/2;0,1;-sqrt(2)/2,sqrt(2)/2;-1,0;-sqrt(2)/2,-sqrt(2)/2;0,-1;sqrt(2)/2,-sqrt(2)/2;0,0];
    step_sizes = [0.2,0.1,0.05,0.02,0.01,0.008];
    
    % Algorithm
    
    % Start timers
    Total_time = tic;
    Init = tic;
    Loop_time = tic;
    
    if sP == 1
        fprintf('Initializing... \n')
    end
    
    % define minimum distance between points
    if n > 99
        min_dist = 1.5*sqrt(1/(pi*n));
    elseif n>4
        min_dist = 2*sqrt(1/(pi*n));
    else
        min_dist = 3*sqrt(1/(pi*n));
    end
    
    % Initial points solution
    [d_old, x_old, y_old] = Initialization(n, cfun, min_dist);
    
    Init_time = toc(Init);
    if sP == 1
        fprintf('Initializing time: %f \n', Init_time)
    end
    
    % Initialize optimum
    d_opt = d_old;
    x_opt = x_old;
    y_opt = y_old;
    
    % Drawing the moving plot
    if sP == 1
        figure;
        drawContainer(cfun)
        hold on
        h_points = plot(x_old, y_old, 'r.', 'MarkerSize', 15);
    end
    
    % Initialize iteration counter, tabu list and tabu list counter
    iteration = 1;
    tabu_list = zeros(4, tabu_len);
    t =1;
    
    % Intensification initials
    it_without_improv = 0;
    nm_intensification = 0;
    no_improv = tic;
    intens_it = [];
    last_int = 0;
    diversification = 0;
    
    % Aspiration initials
    nm_aspiration_considered = 0;
    nm_aspiration_used = 0;
    asp_it_considered = [];
    asp_it_used = [];
    
    % Stopping criterion initial
    d_opt_before_intens = 0;

    % Show progress time start
    print_time = tic;
    
    % Initialize array for storing objective value for plot
    ds = [];

    if sP == 1
        fprintf('Begin Tabu Search \n')
    end
    
    while toc(Loop_time) < T
        
         
        if sP == 1
            % Determine the distances between points
            current_points = [x_old, y_old];
            dist = pdist(current_points);
            % Determine the minimum distance and the pair of points
            [min_d, ind]= min(dist);
            sub = ind_2_sub_tri(n, ind);
            % Plots line between two closest points 
            plot_line = line([x_old(sub(1)), x_old(sub(2))], [y_old(sub(1)), y_old(sub(2))], 'Color', 'black', 'LineWidth', 0.5);
        end
        
        % Access the neighbourhood of moves of the closest two points
        [d_new, x_new, y_new, numbers, directions, Aspiration_flag1, Aspiration_flag2] = neighbourhood(n,x_old, y_old, cfun, tabu_list, d_opt, point_directions, step_sizes);
        

        % Counters, flags and iteration arrays used to show when aspiration
        % was considered, used and at which iterations.
        if Aspiration_flag2 == 1
            nm_aspiration_considered = nm_aspiration_considered + Aspiration_flag1;
            nm_aspiration_used = nm_aspiration_used + Aspiration_flag2;
            asp_it_considered = [asp_it_considered, iteration];
            asp_it_used = [asp_it_used, iteration];
        elseif Aspiration_flag1 == 1
            nm_aspiration_considered = nm_aspiration_considered + Aspiration_flag1;
            asp_it_considered = [asp_it_considered, iteration];
        end
       
        
        % Save the best solution
        if d_new > d_opt
            d_opt = d_new;
            x_opt = x_new;
            y_opt = y_new;
            it_without_improv = 0;
            no_improv = tic;
            opt_it = iteration;
        end
    
        % Store the solution 
        ds = [ds,d_new];
        d_old = d_new;
        x_old = x_new;
        y_old = y_new;
        iteration = iteration + 1;
        it_without_improv = it_without_improv +1;
        
    
        % Generating the tabu list: consists of opposite directions
        % For all pairs of points moved
        opposite_dir = zeros(1,2);
        for h = 1:2
            % Find tabu direction
            if directions(h) == 9
                opposite_dir(h) = 9;
            elseif directions(h) <5
                opposite_dir(h) = directions(h) + 4;
            else
                opposite_dir(h) = directions(h) - 4;
            end
            
            if opposite_dir(h) == 9
                neighbors = [9;9;9];
            elseif opposite_dir(h) == 1
                neighbors = [8;1;2];
            elseif opposite_dir(h) == 8
                neighbors = [7;8;1];
            else
                neighbors = [opposite_dir(h)-1; opposite_dir(h); opposite_dir(h) + 1];
            end
            
            % Update tabu list
            tabu_list(:,t+h-1) = [numbers(1,h); neighbors];
        end
    
        % Update tabu counter
        t = t +2;
        % if t == length(tabu_list) +1 % should be t > length(tabu_list)
        if t > length(tabu_list)
            t =1;
        end
        
        % Update the moving plot
        if sP == 1
            set(h_points, 'XData', x_new, 'YData', y_new);
            drawnow
            delete(plot_line);
        end
        
        % Iteration print
        if toc(print_time) > 2 && sP == 1
            fprintf('Iteration number: %i        Lower Bound of Objective value: %f \n', iteration, d_opt)
            print_time = tic;
        end
        
    
        % Diversification
        if toc(no_improv) > max_time_no_improv && diversification == 0 && toc(Loop_time) < 0.5*T       % no improv & not happend yet & still enough time
            min_dist = d_opt;
            [d_old, x_old, y_old] = Initialization(n, cfun, min_dist);  
            if sP == 1
                fprintf('Diversification occurred \n')
            end
    
            % Reset
            it_without_improv = 0;
            no_improv = tic;
            tabu_list = zeros(4, tabu_len);
            diversification = 1;
            divers_it = iteration;
    
            % Save if the best solution
            if d_new > d_opt
                d_opt = d_new;
                x_opt = x_new;
                y_opt = y_new;
                it_without_improv = 0;
                no_improv = tic;
                opt_it = iteration;
            end
        
        % Intensification, go back to optimum with clean tabu list
        elseif toc(no_improv) > max_time_no_improv && last_int == 0
            if sP == 1
                fprintf('Intensification occurred \n')
            end
            
            % Stopping criterion
            if d_opt_before_intens == d_opt
                break
            end
            
            % reset the stopping criterion
            d_opt_before_intens = d_opt;
            
            % Return the algorithm back to the best found solution and
            % clear the tabu list. Also reset the iteration counter and
            % time counter.
            d_old = d_opt;
            x_old = x_opt;
            y_old = y_opt;
            tabu_list = zeros(4, tabu_len); 
            nm_intensification = nm_intensification +1;
            it_without_improv = 0;
            no_improv = tic;
            intens_it = [intens_it, iteration];
            % Change the step_sizes to smaller scale to search more
            % intensively
            step_sizes = [0.05,0.02,0.01,0.005,0.001]; 


    
        % Last part always try intensification
        elseif toc(Loop_time)> 0.9*T && last_int == 0
            if sP == 1
                fprintf('End Intensification occurred \n')
            end

            % Return the algorithm back to the best found solution and
            % clear the tabu list. Also reset the iteration counter and time 
            % counter.
            d_old = d_opt;
            x_old = x_opt;
            y_old = y_opt;
            tabu_list = zeros(4, tabu_len); 
            nm_intensification = nm_intensification +1;
            it_without_improv = 0;
            no_improv = tic;
            last_int = 1;
            intens_it = [intens_it, iteration];
            % Change the step_sizes to smaller scale to search more
            % intensively
            step_sizes = [0.02,0.01,0.005,0.001];
        end
    
    end
    % End timer
    time=toc(Total_time);

    % Objective value plot
    figure;
    plot(1:iteration-1,ds)
    title('Objective value per iteration plot')
    
    % Final plot
    figure;
    drawContainer(cfun)
    hold on
    plot(x_opt, y_opt, 'r.', 'MarkerSize', 15);
    title('Best found plot')


    % Print summary of diversification, intensification, aspiration and
    % best solution.
    if diversification ==0
        disp('No diversification')
    else
        fprintf('Diversification at iteration: %i\n', divers_it)
    end
    
    fmt1 = ['Number of times intensification: %i at iterations' repmat(' %1.0f,',1,numel(intens_it)) '\n'];
    fprintf(fmt1, nm_intensification, intens_it)
    fmt2 = ['Number of times aspiration considered: %i at iterations' repmat(' %1.0f,',1,numel(asp_it_considered)) '\n'];
    fprintf(fmt2, nm_aspiration_considered, asp_it_considered)
    fmt3 = ['Number of times aspiration used: %i at iterations' repmat(' %1.0f,',1,numel(asp_it_used)) '\n'];
    fprintf(fmt3, nm_aspiration_used, asp_it_used)
    fprintf('Best found minimum distance: %f at iteration %i\n', d_opt, opt_it )
    
    
    % Plot line between closest points
    
    % Determine the distances between points
    opt_points = [x_opt, y_opt];
    opt_dist = pdist(opt_points);
    % Determine the minimum distance and the pair of points
    [~, ind]= min(opt_dist);
    sub = ind_2_sub_tri(n, ind);
    % Plots line between two closest points 
    plot_line = line([x_opt(sub(1)), x_opt(sub(2))], [y_opt(sub(1)), y_opt(sub(2))], 'Color', 'black', 'LineWidth', 0.5);
    fprintf('time taken: %f \n', time)

    % Saving the final best solution
    d = d_opt;
    x = x_opt;
    y = y_opt;

end

function [min_dist,x_old,y_old] = Initialization(n,cfun, min_dist)


    % Define counter for how many actually get placed
    num_points = 0;
    % Define number of attempts before reducing distance
    max_num_iter = 1000 * n;
    count = 1;
    % Create arrays for the x and y coordinates
    x = zeros(1, n);
    y = zeros(1, n);
    num_per_try = 0;
    total_times = 0;
    
    while num_points < n
        % Get a random point
        xPossible = rand();
        yPossible = rand();
        total_times = total_times + 1;
    
        % Check if in figure
        if cfun(xPossible, yPossible) ~= 1
            continue
        end
    
        if num_points == 0
            % First point automatically is valid if it is in the container
            num_points = num_points + 1;
            x(num_points) = xPossible;
            y(num_points) = yPossible;
            continue;
        end
    
        % Find distances between this point and all others
        distances = sqrt((x-xPossible) .^ 2 + (y - yPossible) .^ 2);
        if min(distances) >= min_dist
            % If far enough away from all the other points, add it 
            num_points = num_points + 1;
            x(num_points) = xPossible;
            y(num_points) = yPossible;
        end
        % Increase the loop count
        count = count + 1;
        
        % Check if max is reached
        if count == max_num_iter
            num_this_attempt = num_points - num_per_try(end);
            num_per_try = [num_per_try num_this_attempt];
            
            % Decrease minimum distance
            min_dist = 0.99 * min_dist;
            
            % Reset all counters and lists, if want to start over
            count = 0;
            num_points = 0;
            x = zeros(1, n);
            y = zeros(1, n);
        end
    end
    x_old = x';
    y_old = y';
end

function sub = ind_2_sub_tri(n, ind)
    % Function that takes the number of points n, and the index of the
    % minimum distance from the distance matrix and returns the
    % corresponding pair of points sub.
    list = cumsum(n-1:-1:1);
    index = find(list>=ind,1);
    if index > 1
        excess = ind - list(index -1);
    else
        excess = ind;
    end
    column = index + excess;
    sub = [index,column];
end

function [min_d, x, y, numbers, directions, Aspiration_flag1, Aspiration_flag2] = neighbourhood(n, x, y, cfun, tabu_list, d_opt, point_directions, step_sizes)

    % Determine the distances between points
    points = [x, y];
    dist = pdist(points);
    
    % Determine the minimum distance and the pair of points
    [~, ind]= min(dist);
    sub = ind_2_sub_tri(n, ind);
    numbers = reshape(sub', 1,2);
    
    % Move directions (unit vectors) and step sizes
    num_dir = size(point_directions,1);
    num_steps = size(step_sizes,2);
    
    % Check for tabu
    check = ismember(numbers(1:2), tabu_list(1,:));
    [~, ~, uniqueIndices] = unique(tabu_list(1,:));
    max_count = max(accumarray(uniqueIndices, 1));
    tabu = zeros(max_count*3,2);
    for i = 1:2
        if check(i) ==1
            index = find(numbers(i) == tabu_list(1,:));
            if size(index,2) == 1
                tabu(1:3,i) = tabu_list(2:4, check(i)); %  tabu(1:3,i) = tabu_list(2:4, index); ??
            else
                num_of_occ = 3*size(index, 2);
                tabu(1:num_of_occ,i) = reshape(tabu_list(2:4,index), num_of_occ, 1);
            end
        end
    end
    
    % Create new empty matrix to store new points
    new_xy = zeros(num_dir^2*num_steps,6);
    tabu_xy = zeros(num_dir^2*num_steps,6);
    
    % For loop that stores new_points if they are feasible (inside container)
    counter = 1;
    tabu_counter = 1;
    for i=1:num_dir
        for p = 1:num_dir
            for j =1:num_steps
                loc = 1:4;
                % Sets step_size
                step_size = step_sizes(j);
                % Sets the directions for the two points
                v1 = point_directions(i,:)*step_size;
                v2 = point_directions(p,:)*step_size;
                
                % Both tabu
                if ismember(i, tabu(:,1)) == 1  && ismember(p, tabu(:,2)) == 1
                    tabu_xy(tabu_counter,loc) = [x(numbers(1))+v1(1), y(numbers(1)) + v1(2), x(numbers(2))+v2(1), y(numbers(2)) + v2(2)];
                    tabu_xy(tabu_counter, 5) = i;
                    tabu_xy(tabu_counter, 6) = p;
                    tabu_counter = tabu_counter + 1;
    
                    new_xy(counter,:) = [];
                    continue

                % First point tabu, second point not tabu
                elseif ismember(i, tabu(:,1)) == 1  && ismember(p, tabu(:,2)) == 0
                    tabu_xy(tabu_counter,loc) = [x(numbers(1))+v1(1), y(numbers(1)) + v1(2), x(numbers(2))+v2(1), y(numbers(2)) + v2(2)];
                    tabu_xy(tabu_counter, 5) = i;
                    tabu_xy(tabu_counter, 6) = p;
                    tabu_counter = tabu_counter + 1;
    
                    new_xy(counter,loc) = [x(numbers(1)), y(numbers(1)), x(numbers(2))+v2(1), y(numbers(2)) + v2(2)];
                    % Keep track of direction for tabu list
                    new_xy(counter,5) = 9;
                    new_xy(counter,6) = p;

                    % At least one point has to move
                    if all(point_directions(p,:) == [0,0])
                        new_xy(counter,:) = []; 
                        continue
                    end
                
                % First point not tabu, second point is tabu
                elseif ismember(i, tabu(:,1)) == 0  && ismember(p, tabu(:,2)) == 1
                    tabu_xy(tabu_counter,loc) = [x(numbers(1))+v1(1), y(numbers(1)) + v1(2), x(numbers(2))+v2(1), y(numbers(2)) + v2(2)];
                    tabu_xy(tabu_counter, 5) = i;
                    tabu_xy(tabu_counter, 6) = p;
                    tabu_counter = tabu_counter + 1;
    
                    new_xy(counter,loc) = [x(numbers(1))+v1(1), y(numbers(1)) + v1(2), x(numbers(2)), y(numbers(2))];
                    % Keep track of direction for tabu list
                    new_xy(counter,5) = i;
                    new_xy(counter,6) = 9;

                    % At least one point has to move
                    if all(point_directions(i,:) == [0,0])
                        new_xy(counter,:) = []; 
                        continue
                    end
                
                % Neither point point
                else
                    new_xy(counter,loc) = [x(numbers(1))+v1(1), y(numbers(1)) + v1(2), x(numbers(2))+v2(1), y(numbers(2)) + v2(2)];
                    
                    % Keep track of direction for tabu list
                    new_xy(counter,5) = i;
                    new_xy(counter,6) = p;
                end
    
    
                if cfun(new_xy(counter,1),new_xy(counter,2)) == 0 % Checks if first new point is feasible
                   new_xy(counter,:) = [];
                   continue
                elseif cfun(new_xy(counter,3),new_xy(counter,4)) == 0 % Checks if second new point is feasible
                   new_xy(counter,:) = []; 
                   continue
                elseif all(point_directions(i,:) == [0,0]) && all(point_directions(p,:) == [0,0]) % At least one point has to move
                   new_xy(counter,:) = []; 
                   continue
                end
                counter = counter + 1;
            end
        end
    end
   
    
    % Cleaning the tabu moves matrix tabu_xy
    row = zeros(1,6);
    empty_rows = ismember(tabu_xy,row, 'row');
    tabu_xy(empty_rows,:) = [];
    feasible_rows = cfun(tabu_xy(:,1),tabu_xy(:,2)) & cfun(tabu_xy(:,3), tabu_xy(:,4));
    tabu_xy(~feasible_rows, :) = [];
    
    if isempty(new_xy) && isempty(tabu_xy)
        disp('No feasible moves \n')
    else 
        
        % Creates empty matrix to store new minimum distances for each new set of
        % points in the neighbourhood
        tabu_min_ds = zeros(size(tabu_xy,1),3);
        
        % For loop that iterates through the neighbourhood of new points
        for f=1:size(tabu_min_ds,1)
            
            new_points = points;
    
            % Saves new coordinates of first point
            new_points(numbers(1),:) = tabu_xy(f,1:2);
            % Saves new coordinates of second point
            new_points(numbers(2),:) = tabu_xy(f,3:4);
    
            % Calculates the new distances
            tabu_dist = pdist(new_points);
            % Finds the new minimum distance
            [tabu_min_d,] = min(tabu_dist);
            % Stores the new minimum distance and the moving directions in the matrix
            tabu_min_ds(f,1) = tabu_min_d;
            tabu_min_ds(f,2) = tabu_xy(f,5);
            tabu_min_ds(f,3) = tabu_xy(f,6);
        end
        
        
        % Finds the solution in the neighbourhood with the maximum
        [min_d_tabu, ind_tabu] = max(tabu_min_ds(:,1));
        

        if isempty(new_xy) % Bug Handling
            
            min_d = min_d_tabu;
            points(numbers(1),:) = tabu_xy(ind_tabu,1:2);
            points(numbers(2),:) = tabu_xy(ind_tabu,3:4);
            x = points(:,1);
            y = points(:,2);
            
            % Save the direction in which the points movedgood_numbers = 1:10
            directions = tabu_min_ds(ind_tabu, 2:3);

            % Indicates that Aspiration criterion was NOT considered nor
            % USED
            Aspiration_flag1 = 0;
            Aspiration_flag2 = 0;
            return
        elseif min_d_tabu > d_opt % Aspiration criterion check
            Aspiration_flag1 = 1; % Indicates that Aspiration criterion was considered
            
            % Creates empty matrix to store new minimum distances for each new set of
            % points in the neighbourhood
            new_min_ds = zeros(size(new_xy,1),3);
            
            % For loop that iterates through the neighbourhood of new points
            for f=1:size(new_min_ds,1)
                
                new_points = points;
        
                % Saves new coordinates of first point
                new_points(numbers(1),:) = new_xy(f,1:2);
                % Saves new coordinates of second point
                new_points(numbers(2),:) = new_xy(f,3:4);
            
                % Calculates the new distances
                new_dist = pdist(new_points);
                % Finds the new minimum distance
                [new_min_d,] = min(new_dist);
                % Stores the new minimum distance and the moving directions in the matrix
                new_min_ds(f,1) = new_min_d;
                new_min_ds(f,2) = new_xy(f,5);
                new_min_ds(f,3) = new_xy(f,6);
            end
            
            % Adding the aspiration move to the new_min_ds and new_xy
            new_min_ds = [new_min_ds; tabu_min_ds(ind_tabu,:)];
            aspiration_xy = tabu_xy(ind_tabu,:);
            new_xy = [new_xy; aspiration_xy];

            % Finds the solution in the neighbourhood with the maximum
            [min_d_neighbourhood, ind_neighbourhood] = max(new_min_ds(:,1));
            if ind_neighbourhood == size(new_min_ds,1)
                Aspiration_flag2 = 1; % Indicates that aspiration criterion was USED
            else
                Aspiration_flag2 = 0; % Indicates that aspiration criterion was NOT USED
            end
            
            min_d = min_d_neighbourhood;
            % Update the points
            points(numbers(1),:) = new_xy(ind_neighbourhood,1:2);
            points(numbers(2),:) = new_xy(ind_neighbourhood,3:4);
            x = points(:,1);
            y = points(:,2);
            
            % Save the direction in which the points moved
            directions = new_min_ds(ind_neighbourhood, 2:3);
            return

        else       % Standard version 
        
            % Creates empty matrix to store new minimum distances for each new set of
            % points in the neighbourhood
            new_min_ds = zeros(size(new_xy,1),3);
            
            % For loop that iterates through the neighbourhood of new points
            for f=1:size(new_min_ds,1)
                
                new_points = points;
        
                % Saves new coordinates of first point
                new_points(numbers(1),:) = new_xy(f,1:2);
                % Saves new coordinates of second point
                new_points(numbers(2),:) = new_xy(f,3:4);
            
                % Calculates the new distances
                new_dist = pdist(new_points);
                % Finds the new minimum distance
                [new_min_d,] = min(new_dist);
                % Stores the new minimum distance and the moving directions in the matrix
                new_min_ds(f,1) = new_min_d;
                new_min_ds(f,2) = new_xy(f,5);
                new_min_ds(f,3) = new_xy(f,6);
            end
            
            % Finds the solution in the neighbourhood with the maximum
            [min_d_neighbourhood, ind_neighbourhood] = max(new_min_ds(:,1));
            
            min_d = min_d_neighbourhood;
            % Update the points
            points(numbers(1),:) = new_xy(ind_neighbourhood,1:2);
            points(numbers(2),:) = new_xy(ind_neighbourhood,3:4);
            x = points(:,1);
            y = points(:,2);
            
            % Saves the direction which the points moved
            directions = new_min_ds(ind_neighbourhood, 2:3);

            % Indicates that Aspiration criterion was NOT considered nor
            % USED
            Aspiration_flag1 = 0;
            Aspiration_flag2 = 0;
        end
    end
end

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