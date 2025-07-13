% plot NAE and CWE on same plot
clear all
close all

%% load nae and cwe results
nae_disjoint = load('naive_disjoint_results.mat');
nae_joint = load('naive_joint_results.mat');
cwe_disjoint = load('disjoint_results.mat');
cwe_joint = load('joint_results.mat');

%% define plot parameters
params_latex = ["\alpha","t_0"];
params_text = ["alpha","t0"];
color_vec = ["#A2142F", "#0072BD", "#333333", "#EDB120", "#77AC30", "#7E2F8E"];


folder_path = "C:\Users\Vincent Huynh\Desktop\Final AirComp Results\";
desktop_path = "C:\Users\Vincent Huynh\Desktop";

%% linear plots

close all

% all_disjoint, all_joint
experiment = "all_joint";

if experiment == "all_disjoint"
    exp1 = nae_disjoint;
    exp2 = cwe_disjoint;
else
    exp1 = nae_joint;
    exp2 = cwe_joint;
end

% load variables from result files
agent_db_values = cwe_disjoint.agent_db_values;
sensor_dimension = cwe_disjoint.sensor_dimension;
selected_schemes = cwe_disjoint.selected_schemes;
sensor_vals = cwe_disjoint.sensor_vals;
channel_db_values = cwe_disjoint.channel_db_values;

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

            for count_idx = 1:length(count_vals)
                plot(x_axis_series,(squeeze(exp1.avg_dep_var(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-^','color',color_vec(count_idx),'LineWidth', 1)
                plot(x_axis_series,(squeeze(exp1.avg_dep_mse(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-x','color',color_vec(count_idx),'LineWidth', 1)
                plot(x_axis_series,(squeeze(exp1.avg_dep_crlb(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'--o','color',color_vec(count_idx),'LineWidth', 1)

                plot(x_axis_series,(squeeze(exp2.avg_dep_var(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-^','color',color_vec(count_idx+3),'LineWidth', 1)
                plot(x_axis_series,(squeeze(exp2.avg_dep_mse(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-x','color',color_vec(count_idx+3),'LineWidth', 1)
                plot(x_axis_series,(squeeze(exp2.avg_dep_crlb(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'--o','color',color_vec(count_idx+3),'LineWidth', 1)
            end

            if experiment == "agent_server_noise"
                xlabel('$$\Gamma$$','Interpreter','latex')
            else
                xlabel('Channel SNR (dB)')
            end
            ylabel(" ")

            % title('$$\hat{'+params(jj)+'}$$, ' + selected_schemes(scheme_idx),'Interpreter','latex')
            title('$$\hat{'+params_latex(param_idx)+'}$$','Interpreter','latex')
            set(ax, 'FontSize', 15);
            set(ax, 'YScale', 'log')

            if scheme == "MPC"
                legend(["NAE, S = " + sensor_vals(1:end),"CWE, S = " + sensor_vals(1:end),"VAR","MSE","CRLB"], 'Location', 'north east')
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