%%% 01-18-2026 - NOTE: this script is currently deprecated due to updates in the matched filtering code, this should still work with the old version of the alpha_t0.m code (from around Sept. 7, 2025 submission)

% plot perfect tpi and imperfect tpi results
clear all
close all

%% define plot parameters
params_latex = ["\alpha","t_0"];
params_text = ["alpha","t0"];
color_vec = [ ...
    "#A2142F", ... % dark red / maroon
    "#0072BD", ... % blue (MATLAB default blue)
    "#333333", ... % dark gray / charcoal
    "#EDB120", ... % gold / mustard yellow
    "#77AC30", ... % green (MATLAB default green)
    "#7E2F8E"  ... % purple / violet
];

home_dir = char(java.lang.System.getProperty('user.home'));
folder_path = home_dir + "\Desktop\Updated Final AirComp Results\";

%% linear plots

close all

% nae_joint_all_tpi, cwe_joint_all_tpi
experiment = "nae_joint_all_tpi";

if experiment == "nae_all_tpi"
    nae_joint_perf_tpi = load('nae_joint_results.mat');
    nae_joint_imperf_tpi = load('nae_joint_imperfect_tpi_results.mat');

    exp1 = nae_joint_perf_tpi;
    exp2 = nae_joint_imperf_tpi;
else
    cwe_joint_perf_tpi = load('cwe_joint_results.mat');
    cwe_joint_imperf_tpi = load('cwe_joint_imperfect_tpi_results.mat');

    exp1 = cwe_joint_perf_tpi;
    exp2 = cwe_joint_imperf_tpi;
end

% load variables from result files
agent_db_values = exp1.agent_db_values;
sensor_dimension = exp1.sensor_dimension;
selected_schemes = exp1.selected_schemes;
sensor_vals = exp1.sensor_vals;
channel_db_values = exp1.channel_db_values;

for agent_db_idx = 1:length(agent_db_values)
    % create save directory if it does not exist
    save_folder = folder_path + experiment + "_" + string(agent_db_values(agent_db_idx)) + "_db"; % Define save folder
    if ~exist(save_folder, 'dir') % Check if the folder exists
        mkdir(save_folder); % Create the folder if it doesn't exist
    end

    for param_idx = 1:2
        % collect all plot values for y-axis scaling
        all_vals = [exp1.avg_dep_var(:,agent_db_idx,:,param_idx,:,1:sensor_dimension);...
                    exp1.avg_dep_mse(:,agent_db_idx,:,param_idx,:,1:sensor_dimension);...
                    exp1.avg_dep_crlb(:,agent_db_idx,:,param_idx,:,1:sensor_dimension);
                    exp2.avg_dep_var(:,agent_db_idx,:,param_idx,:,1:sensor_dimension);...
                    exp2.avg_dep_mse(:,agent_db_idx,:,param_idx,:,1:sensor_dimension);...
                    exp2.avg_dep_crlb(:,agent_db_idx,:,param_idx,:,1:sensor_dimension)];
        for scheme_idx = 1:length(selected_schemes)
            scheme = selected_schemes(scheme_idx);
            fig1 = figure;
            set(fig1,'Position',[100,100,450,450])
            ax = gca;

            hold on

            x_axis_series = channel_db_values(scheme_idx,:,agent_db_idx);
            count_vals = sensor_vals;
            
            % plot pseudoplots for legend symbols
            for sensor_idx = 1:length(count_vals)
                plot(nan,nan,'color',color_vec(sensor_idx),'LineWidth',2);
                plot(nan,nan,'color',color_vec(sensor_idx+3),'LineWidth',2);
            end

            plot(nan,nan,'^','color','black');
            plot(nan,nan,'x','color','black');
            plot(nan,nan,'o','color','black');
           
            if scheme == "NPC"
                count_vals = count_vals(1);
            end

            legend_entries = [];

            for count_idx = 1:length(count_vals)
                plot(x_axis_series,(squeeze(exp1.avg_dep_var(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-^','color',color_vec(count_idx),'LineWidth', 1)
                plot(x_axis_series,(squeeze(exp1.avg_dep_mse(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-x','color',color_vec(count_idx),'LineWidth', 1)
                plot(x_axis_series,(squeeze(exp1.avg_dep_crlb(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'--o','color',color_vec(count_idx),'LineWidth', 1)
                legend_entries = [legend_entries, "Sync., S = " + count_vals(count_idx)];

                plot(x_axis_series,(squeeze(exp2.avg_dep_var(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-^','color',color_vec(count_idx+3),'LineWidth', 1)
                plot(x_axis_series,(squeeze(exp2.avg_dep_mse(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-x','color',color_vec(count_idx+3),'LineWidth', 1)
                % plot(x_axis_series,(squeeze(exp2.avg_dep_crlb(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'--o','color',color_vec(count_idx+3),'LineWidth', 1)
                legend_entries = [legend_entries, "Unsync., S = " + count_vals(count_idx)];
            end

            
            xlabel('Channel SNR (dB)')
            ylabel(" ")

            % title('$$\hat{'+params(jj)+'}$$, ' + selected_schemes(scheme_idx),'Interpreter','latex')
            title('$$\hat{'+params_latex(param_idx)+'}$$','Interpreter','latex')
            set(ax, 'FontSize', 15);
            set(ax, 'YScale', 'log')

            if scheme == "MPC"
                legend([legend_entries,"VAR","MSE","CRLB"], 'Location', 'north east')
            end

            % Add box around plot axes.
            box(ax, 'on');
            grid on;

            y_lower = 10^(-0.25)*min(all_vals,[],"all");
            y_upper = 1.05*max(all_vals,[],"all");
            ylim([y_lower,y_upper]);
            y_ticks = power_10_range(y_lower,y_upper);

            ax.YTick = y_ticks;
            ax.GridColor = [0 0 0];
            ax.GridAlpha = 0.5;
            ax.LineWidth = 1;

            %%% Save the figure
            exportgraphics(fig1, fullfile(save_folder, sprintf(experiment + "_" + string(agent_db_values(agent_db_idx)) + "_db_" + params_text(param_idx) + "_" + selected_schemes(scheme_idx) + ".png")), 'Resolution', 300);
        end
    end
end