% compare OMA and CWE
clear all
close all

%% define plot parameters
params_latex = ["\alpha","t_0"];
params_text = ["alpha","t0"];

color_vec = [
    "#A2142F"  % dark red
    "#008000"  % green
    "#0072BD"  % blue
    "#D95319"  % orange
    "#4DBEEE"  % light blue (cyan)
    "#7E2F8E"  % purple
];

home_dir = char(java.lang.System.getProperty('user.home'));

%% linear plots
close all

% options: "oma_cwe_disjoint", "oma_cwe_joint"
experiment = "oma_cwe_joint";
disp("Plotting " + experiment)

if experiment == "oma_cwe_disjoint"
    cwe_disjoint = load('comp_cwe_disjoint_results.mat');
    oma_disjoint = load('oma_disjoint_results.mat');

    exp1 = cwe_disjoint;
    exp2 = oma_disjoint;
else
    cwe_joint = load('comp_cwe_joint_results.mat');
    oma_joint = load('oma_joint_results.mat');

    exp1 = cwe_joint;
    exp2 = oma_joint;
end

% load variables from result files
agent_db_values = exp1.agent_db_values;
sensor_dimension = exp1.sensor_dimension;
selected_schemes = exp1.selected_schemes;
sensor_vals = exp1.sensor_vals;
channel_db_values = exp1.channel_db_values;

what_to_plot = "all"; % "all", "EPC";

if what_to_plot == "all"
    selected_scheme_idx = 1:4;
    folder_path = home_dir + "\Desktop\Updated Final AirComp Results\Response Letter\";
elseif what_to_plot == "EPC"
    selected_scheme_idx = 2;
    folder_path = home_dir + "\Desktop\Updated Final AirComp Results\For Manuscript\";
else
    error("Must choose all or EPC")
end

for agent_db_idx = 1:length(agent_db_values)
    % create save directory if it does not exist
    save_folder = folder_path + experiment + "_" + string(agent_db_values(agent_db_idx)) + "_db"; % Define save folder
    if ~exist(save_folder, 'dir') % Check if the folder exists
        mkdir(save_folder); % Create the folder if it doesn't exist
    end

    % [alpha, t0]
    for param_idx = 1:2
        % collect all plot values for y-axis scaling
        all_vals = [exp1.avg_dep_var(:,agent_db_idx,:,param_idx,selected_scheme_idx,1:sensor_dimension);...
                    exp1.avg_dep_mse(:,agent_db_idx,:,param_idx,selected_scheme_idx,1:sensor_dimension);...
                    exp1.avg_dep_crlb(:,agent_db_idx,:,param_idx,selected_scheme_idx,1:sensor_dimension);
                    exp2.avg_dep_var(:,agent_db_idx,:,param_idx,selected_scheme_idx,1:sensor_dimension);...
                    exp2.avg_dep_mse(:,agent_db_idx,:,param_idx,selected_scheme_idx,1:sensor_dimension);...
                    exp2.avg_dep_crlb(:,agent_db_idx,:,param_idx,selected_scheme_idx,1:sensor_dimension)];

        for scheme_idx = selected_scheme_idx % 1:length(selected_schemes)
            scheme = selected_schemes(scheme_idx);
            fig1 = figure;
            set(fig1,'Position',[100,100,450,450])
            ax = gca;

            hold on

            x_axis_series = channel_db_values(scheme_idx,:,agent_db_idx);
            count_vals = sensor_vals;
            
            % plot pseudoplots for legend symbols
            % h_symbols(1) = plot(nan,nan,'-','color','black');
            % h_symbols(2) = plot(nan,nan,'--','color','black');
            % h_symbols(3) = plot(nan,nan,'^','color','black');
            % h_symbols(4) = plot(nan,nan,'x','color','black');
            % h_symbols(5) = plot(nan,nan,'o','color','black');

            h_symbols(1) = plot(nan,nan,'^','color','black');
            h_symbols(2) = plot(nan,nan,'x','color','black');
            h_symbols(3) = plot(nan,nan,'o','color','black');

            for sensor_idx = 1:length(count_vals)
                % h_colors(sensor_idx) = plot(nan,nan,'color',color_vec(sensor_idx),'LineWidth',2);
                h_colors(2*sensor_idx-1) = plot(nan,nan,'color',color_vec(2*sensor_idx-1),'LineWidth',2);
                h_colors(2*sensor_idx) = plot(nan,nan,'color',color_vec(2*sensor_idx),'LineWidth',2);
            end

            if scheme == "NPC"
                count_vals = count_vals(1);
            end

            legend_entries = [];

            for count_idx = 1:length(count_vals)
                % plot(x_axis_series,(squeeze(exp1.avg_dep_var(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-^','color',color_vec(count_idx),'LineWidth', 1)
                % plot(x_axis_series,(squeeze(exp1.avg_dep_mse(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-x','color',color_vec(count_idx),'LineWidth', 1)
                % plot(x_axis_series,(squeeze(exp1.avg_dep_crlb(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-o','color',color_vec(count_idx),'LineWidth', 1)
                % legend_entries = [legend_entries, "CWE, S = " + count_vals(count_idx)];
                % 
                % plot(x_axis_series,(squeeze(exp2.avg_dep_var(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'--^','color',color_vec(count_idx),'LineWidth', 1)
                % plot(x_axis_series,(squeeze(exp2.avg_dep_mse(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'--x','color',color_vec(count_idx),'LineWidth', 1)
                % plot(x_axis_series,(squeeze(exp2.avg_dep_crlb(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'--o','color',color_vec(count_idx),'LineWidth', 1)
                % legend_entries = [legend_entries, "OMA, S = " + count_vals(count_idx)];

                plot(x_axis_series,(squeeze(exp1.avg_dep_var(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-^','color',color_vec(2*count_idx-1),'LineWidth', 1)
                plot(x_axis_series,(squeeze(exp1.avg_dep_mse(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-x','color',color_vec(2*count_idx-1),'LineWidth', 1)
                plot(x_axis_series,(squeeze(exp1.avg_dep_crlb(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-o','color',color_vec(2*count_idx-1),'LineWidth', 1)
                legend_entries = [legend_entries, "CWE, S = " + count_vals(count_idx)];

                plot(x_axis_series,(squeeze(exp2.avg_dep_var(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'--^','color',color_vec(2*count_idx),'LineWidth', 1)
                plot(x_axis_series,(squeeze(exp2.avg_dep_mse(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'--x','color',color_vec(2*count_idx),'LineWidth', 1)
                plot(x_axis_series,(squeeze(exp2.avg_dep_crlb(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'--o','color',color_vec(2*count_idx),'LineWidth', 1)
                legend_entries = [legend_entries, "OMA, S = " + count_vals(count_idx)];
            end

            ylabel(" ")

            % title('$$\hat{'+params(jj)+'}$$, ' + selected_schemes(scheme_idx),'Interpreter','latex')
            title('$$\hat{'+params_latex(param_idx)+'}$$','Interpreter','latex')
            set(ax, 'FontSize', 15);
            set(ax, 'YScale', 'log')

            % Add box around plot axes.
            box(ax, 'on');
            grid on;
            
            if param_idx == 1
                y_lower = 10^(-0.25)*min(all_vals,[],"all");
                y_upper = 2*max(all_vals,[],"all");
            else
                y_lower = 10^(-0.25)*min(all_vals,[],"all");
                y_upper = 25*max(all_vals,[],"all");
            end
            
            ylim([y_lower,y_upper]);
            y_ticks = power_10_range(y_lower,y_upper);

            ax.YTick = y_ticks;
            ax.GridColor = [0 0 0];
            ax.GridAlpha = 0.5;
            ax.LineWidth = 1;

            if (what_to_plot == "all" && scheme == "MPC") || (what_to_plot == "EPC" && scheme == "EPC") 
                
            
                lgd1 = legend(h_colors, cellstr(legend_entries), 'Location', 'north east');
                lgd1_pos = lgd1.Position;
            
                ax2 = axes('Position', ax.Position, 'Color', 'none', 'XTick', [], 'YTick', [], 'Box', 'off');
                
                % lgd_sensors = "S=" + count_vals;
                % lgd2 = legend(ax2, h_colors, cellstr(lgd_sensors), 'Location', 'north west', 'Color', 'white', 'FontSize', 15);
                lgd2 = legend(ax2, h_symbols, cellstr(["VAR","MSE","CRLB"]), 'Location', 'north west', 'Color', 'white', 'FontSize', 15);
                
                lgd2_pos = lgd2.Position;

                gap = 0.018;
                new_x = lgd1_pos(1) - lgd2_pos(3) - gap;
                new_y = lgd1_pos(2) + lgd1_pos(4) - lgd2_pos(4) + 0.001;
                % [x, y, width, height]
                lgd2.Position = [new_x new_y lgd2_pos(3) lgd2_pos(4)];
            end

            %%% Save the figure
            exportgraphics(fig1, fullfile(save_folder, sprintf(experiment + "_" + string(agent_db_values(agent_db_idx)) + "_db_" + params_text(param_idx) + "_" + selected_schemes(scheme_idx) + ".png")), 'Resolution', 300);
        end
    end
end