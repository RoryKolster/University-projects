clc
clear
close all

%% Data set and method choice:
% For data, choose for dataNum 1-4 where 1,2,3,4 is UK_wave1, UK_wave2, NL_wave1, NL_wave2
% respectively.
% For method, choose 1 for Newton method, and 2 for Gauss-Newton
for j=1:4
    for i=1:2
        dataNum = j;
        method = i;
        
        
        %% Rest of code
        opts = detectImportOptions('total_deaths_per_million.csv');
        opts.VariableNamingRule = 'preserve';
        table = readtable('total_deaths_per_million.csv', opts);
        UK = table2array(table(:,235));
        NL = table2array(table(:, 156));
        daily = 0;
        
        
        if daily == 1
            UK_wave1 = UK(29:259);
            UK_wave2 = UK(183:546);
            NL_wave1 = NL(64:266);
            NL_wave2 = NL(169:616);
            time_cell = {29:259, 183:546, 64:266, 169:616};
        else
            UK_wave1 = UK(35:7:259);
            UK_wave2 = UK(189:7:546);
            NL_wave1 = NL(70:7:266);
            NL_wave2 = NL(175:7:616);
            time_cell_ = {35:7:259, 189:7:546, 70:7:266, 175:7:616};
            time_cell = {1:7:231, 1:7:364, 1:7:203, 1:7:448};
        end
        data_cell = {UK_wave1, UK_wave2, NL_wave1, NL_wave2};
        
        
        
        % Logistic function
        phi = @(t, a, K, P_0, v) (K) ./ (1 + ((K - P_0)/ P_0) * exp(-a * t)) + v;
        % First order derivatives
        Dr = @(t, a, K, P_0) -(K.*P_0.*(P_0-K).*t.*exp(t.*a))./(P_0.*exp(t.*a)-P_0+K).^2;
        DK = @(t, a, K, P_0) (P_0.^2.*(exp(a.*t)-1).*exp(a.*t))./(K+P_0.*exp(a.*t)-P_0).^2;
        DP_0 = @(t, a, K, P_0) (K.^2.*exp(a.*t))./((exp(a.*t)-1).*P_0+K).^2;
        % Second order derivatives
        Dr2 = @(t, a, K, P_0) (K.*P_0.*(P_0-K).*t.^2.*exp(t.*a).*(P_0.*exp(t.*a)+P_0-K))./(P_0.*exp(t.*a)-P_0+K).^3; 
        DrDK = @(t, a, K, P_0) (P_0.^2.*t.*exp(a.*t).*((2.*exp(a.*t)-1).*K-P_0.*exp(a.*t)+P_0))./(K+P_0.*exp(a.*t)-P_0).^3; 
        DrDP_0 = @(t, a, K, P_0) -(K.^2.*t.*exp(a.*t).*((exp(a.*t)+1).*P_0-K))./((exp(a.*t)-1).*P_0+K).^3; 
        DK2 = @(t, a, K, P_0) -(2.*P_0.^2.*(exp(a.*t)-1).*exp(a.*t))./(K+P_0*exp(a.*t)-P_0).^3; 
        DP_0DK = @(t, a, K, P_0) (2.*P_0.*(exp(a.*t)-1).*exp(a.*t).*K)./(K+P_0.*exp(a.*t)-P_0).^3; 
        DP_02 = @(t, a, K, P_0) -(2.*K.^2.*(exp(a.*t)-1).*exp(a.*t))./((exp(a.*t)-1).*P_0+K).^3; 
        
        % Jacobian, nabla_r, and nabla2_r
        Jac = @(t, a, K, P_0) [-Dr(t, a, K, P_0)', -DK(t, a, K, P_0)', -DP_0(t, a, K, P_0)'];
        r_j = @(y, t, a, K, P_0,v) y - phi(t,a,K,P_0,v);
        nabla2_r_j = @(t, a, K, P_0) [-Dr2(t,a,K,P_0), -DrDK(t,a,K,P_0), -DrDP_0(t,a,K,P_0); -DrDK(t,a,K,P_0), -DK2(t,a,K,P_0), -DP_0DK(t,a,K,P_0); -DrDP_0(t,a,K,P_0), -DP_0DK(t,a,K,P_0), -DP_02(t,a,K,P_0)];
        
        
        
        
        
        
        
        % Data-specific parameters for starting:
        time = time_cell{dataNum};
        data = data_cell{dataNum};
        v = data(1);
        x_k = [0; data(end)-v; 1];
        
        m = length(time);
        time_m = time(floor(m/2));
        median_ind = floor(m/2);
        median = data(median_ind);
        a = (1/time_m)*log(((x_k(2) - x_k(3))* (median - v))/((x_k(2) - median + v) * x_k(3)));
        x_k(1) = a;
        
        % Algorithm-specific paramters:
        tolerance = 0.1;
        % line_search = 0;
        % alpha = 0.2;
        % beta = 0.5;
        
        title_strings = {'Plots of UK wave 1', 'Plots UK of wave 2', 'Plots of NL wave 1', 'Plots of NL wave 2'};
        legend_strings = {'UK wave 1', 'UK wave 2', 'NL wave 1', 'NL wave 2'};
        
        
        % subplot(2,2,i);
        % plot(time, data)
        % 
        % initial = @(time) phi(time, x_k(1), x_k(2), x_k(3), v);
        % title(title_strings{i})
        % xlabel('time')
        % ylabel('y')
        % hold on
        % plot(time, initial(time))
        
        
        
        
        % subplot(2,2,2);
        % plot(time, phi(time, x_k(1), x_k(2), x_k(3), v), 'red')
        % title('Model convergence')
        % xlabel('time')
        % ylabel('y')
        % hold on
        % plot(time, initial(time))
        residual = data - phi(time, x_k(1), x_k(2), x_k(3), v)';
        objective = 0.5*sum(residual.^2);
        
        
        
        parameters = [];
        counter = 0;
        while 1
            counter = counter + 1;
            % Saving each iteration of parameters
            parameters = [parameters, x_k];
            if method == 1
                [x_k, singular_check] = N_iteration(time, data, x_k, v, phi, Jac, r_j, nabla2_r_j, tolerance);
            elseif method == 2 && j==3
                break
            else
                [x_k, singular_check] = GN_iteration(time, data, x_k, v, phi, Jac);
            end
            residual = data - phi(time, x_k(1), x_k(2), x_k(3), v)';
            objective_residual = 0.5*sum(residual.^2);
            objective = [objective, objective_residual];
            change = abs((objective(counter+1) - objective(counter)) / objective(counter));
            if counter ~= 1 && change < 10^-4
                fprintf('Number of iterations: %i \n', counter)
                fprintf('Objective value: %f \n', objective(end))
                break
            elseif counter >= 50
                fprintf('Exceeded 50 iterations \n')
                break
            elseif any(isnan(singular_check), 'all')
                fprintf('Singular Hessian approximation at iteration %i', counter)
                break
            % elseif change > 10^3 
                % fprintf('Divergent case at iteration %i', counter)
                % break
            end
            
            % hold on
            % plot(time, phi(time, x_k(1), x_k(2), x_k(3), v))
        
        end
        
        
        %residuals
        % subplot(2,2,3);
        % r_xk = data - phi(time, x_k(1), x_k(2), x_k(3), v)';
        % histogram(r_xk, 12)
        % title('Residuals histogram')
        % xlabel('Residuals')
        % ylabel('frequency')
        % subplot(2,2,4);

        subplot(2,2,j)
        hold on
        plot(1:counter+1, objective)
        title(legend_strings{j})
        xlabel('Iteration')
        ylabel('Objective value')
        
        
    
    
        % Final plot
        % hold on
        % plot(time, data)
        % title(title_strings{i})
        % xlabel('time')
        % ylabel('y')
        
        % final = @(time) phi(time, x_k(1), x_k(2), x_k(3), v);
        % hold on
        % plot(time, final(time))
        % legend({legend_strings{i}, 'initial plot', 'best fit'}, 'Location', 'northwest')
        
        % Plot showing quadratic convergence
        % distances = parameters - repmat(x_k,1,length(parameters));
        % norms = zeros(1, length(parameters));
        % for i=1:length(parameters)
        %     norms(i) = norm(distances(:,i),2);
        % end
        % figure;
        % plot(1:counter, norms)
        % title('Distance to x* at each iteration')
        % xlabel('Iteration')
        % ylabel('Distance to x*')
    end
    hold on
    legend({'Newton method', 'Gauss-Newton method'}, 'Location', 'northeast')

end

function [x_k1, singular_check] = GN_iteration(t, y, x_k, v, phi, J)

    r_xk = y - phi(t, x_k(1), x_k(2), x_k(3), v)';

    singular_check = inv(J(t, x_k(1), x_k(2), x_k(3))' * J(t, x_k(1), x_k(2), x_k(3)));
    singular_check2 = cond(J(t, x_k(1), x_k(2), x_k(3))' * J(t, x_k(1), x_k(2), x_k(3)));

    x_k1 = x_k - (J(t, x_k(1), x_k(2), x_k(3))' * J(t, x_k(1), x_k(2), x_k(3))) \ (J(t, x_k(1), x_k(2), x_k(3))' * r_xk);

end

function [x_k1, singular_check] = N_iteration(t, y, x_k, v, phi, J, r_j, nabla2_r_j, tolerance)

    r_xk = y - phi(t, x_k(1), x_k(2), x_k(3), v)';
    rr_xk = kron(eye(3), r_xk);
    
    nabla_f_sum = zeros(3,3);
    for i=1:length(t)
         nabla_f_sum = nabla_f_sum + r_j(y(i), t(i), x_k(1), x_k(2), x_k(3), v).*nabla2_r_j(t(i), x_k(1), x_k(2), x_k(3));
    end

    nabla_f = J(t, x_k(1), x_k(2), x_k(3))' * r_xk;
    % nabla2_f = inv(J(t, x_k(1), x_k(2), x_k(3))' * J(t, x_k(1), x_k(2), x_k(3)) + H(t, x_k(1), x_k(2), x_k(3))'*rr_xk);
    nabla2_f = inv(J(t, x_k(1), x_k(2), x_k(3))' * J(t, x_k(1), x_k(2), x_k(3)) + nabla_f_sum);
    singular_check = cond(nabla2_f);
    decrement = nabla_f' / (nabla2_f) * nabla_f;
    if abs(decrement/2) < tolerance
        x_k1 = x_k;
        return 
    end
    % newton_step1 = - (J(t, x_k(1), x_k(2), x_k(3))' * J(t, x_k(1), x_k(2), x_k(3)) + H(t, x_k(1), x_k(2), x_k(3))'*rr_xk) \ nabla_f;
    newton_step = - nabla2_f * nabla_f;
    


    x_k1 = x_k + newton_step;
end