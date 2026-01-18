clear all
close all

% load experiment results
load('random_dropout_results.mat');

params_latex = ["\alpha","t_0"];
params_text = ["alpha","t0"];
color_vec = ["#A2142F", "#0072BD", "#333333"];

home_dir = char(java.lang.System.getProperty('user.home'));

what_to_plot = "EPC";

if what_to_plot == "all"
    selected_scheme_idx = 1:4;
    folder_path = home_dir + "\Desktop\Updated Final AirComp Results\All Schemes\";
elseif what_to_plot == "EPC"
    selected_scheme_idx = 2;
    folder_path = home_dir + "\Desktop\Updated Final AirComp Results\For Manuscript\";
else
    error("Must choose all or EPC")
end

for agent_db_idx = 1:length(agent_db_values)
    % Create save directory if it does not exist.
    save_folder = folder_path + experiment + "_" + string(agent_db_values(agent_db_idx)) + "_db";
    if ~exist(save_folder, 'dir')
        mkdir(save_folder);
    end

    for param_idx = 1:2
        % Collect all plot values for y-axis scaling.
        all_vals = [avg_dep_mse(:,agent_db_idx,:,param_idx,selected_scheme_idx,1:sensor_dimension);...
                    avg_dep_var(:,agent_db_idx,:,param_idx,selected_scheme_idx,1:sensor_dimension);...
                    avg_dep_crlb(:,agent_db_idx,:,param_idx,selected_scheme_idx,1:sensor_dimension)];
        for scheme_idx = selected_scheme_idx % EPC
            scheme = selected_schemes(scheme_idx);
            fig1 = figure;
            set(fig1,'Position',[100,100,450,450])
            ax = gca;
            hold on
            
            % Plot pseudoplots for legend symbols.
            if experiment == "random_dropout"
                for dropout_idx = 1:length(dropout_vals)
                    plot(nan,nan,'color',color_vec(dropout_idx),'LineWidth',2);
                end
            else
                for sensor_idx = 1:length(sensor_vals) 
                    plot(nan,nan,'color',color_vec(sensor_idx),'LineWidth',2);
                end
            end
            plot(nan,nan,'^','color','black'); % VAR
            plot(nan,nan,'x','color','black'); % MSE
            plot(nan,nan,'o','color','black'); % CRLB

            x_axis_series = channel_db_values(scheme_idx,:,agent_db_idx);
            
            if experiment == "random_dropout"
                count_vals = dropout_vals;
            else
                count_vals = sensor_vals;
            end

            if scheme == "NPC"
                count_vals = count_vals(1);
            end

            for count_idx = 1:length(count_vals)
                plot(x_axis_series,(squeeze(avg_dep_var(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-^','color',color_vec(count_idx),'LineWidth', 1)
                plot(x_axis_series,(squeeze(avg_dep_mse(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-x','color',color_vec(count_idx),'LineWidth', 1)
                plot(x_axis_series,(squeeze(avg_dep_crlb(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'--o','color',color_vec(count_idx),'LineWidth', 1)
            end

            xlabel('Channel SNR (dB)')
            ylabel(" ")

            title('$$\hat{'+params_latex(param_idx)+'}$$','Interpreter','latex')
            
            set(ax, 'FontSize', 15);
            set(ax, 'YScale', 'log')

            legend(["Drop = " + dropout_vals,"MSE","VAR","CRLB"], 'Location', 'north east', 'NumColumns', 2)
                
            % Add box around plot axes.
            box(ax, 'on');
            grid on

            if param_idx == 1
                y_lower = 10^(-0.1)*min(all_vals,[],"all");
                y_upper = 1.05*max(all_vals,[],"all");
            else
                y_lower = 10^(-0.25)*min(all_vals,[],"all");
                y_upper = 3.5*max(all_vals,[],"all");
            end
            
            ylim([y_lower,y_upper]);
            
            % y_ticks = power_10_range(y_lower,y_upper);
            % ax.YTick = y_ticks;

            ax.GridColor = [0 0 0];
            ax.GridAlpha = 0.5;
            ax.LineWidth = 1;

            % Save the figure.
            exportgraphics(fig1, fullfile(save_folder, sprintf(experiment + "_" + string(agent_db_values(agent_db_idx)) + "_db_" + params_text(param_idx) + "_" + selected_schemes(scheme_idx) + ".png")), 'Resolution', 300);
        end
    end
end