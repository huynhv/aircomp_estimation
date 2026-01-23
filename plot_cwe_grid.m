close all;
clear all;

% load experiment results
load('cwe_joint_grid_results.mat')

params_latex = ["\alpha","t_0"];
params_text = ["alpha","t0"];

home_dir = char(java.lang.System.getProperty('user.home'));

% Toggle if you want to convert meshgrid values to log (you should).
convert_to_log = true;

% options: "all", "EPC"
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

% Create save directory if it does not exist.
save_folder = folder_path + experiment + "_" + string(agent_db_values(end)) + "_db";
if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end

for param_idx = 1:2
    % Collect all plot values for y-axis scaling.
    all_vals = [avg_dep_mse(:,:,:,param_idx,selected_scheme_idx,1:sensor_dimension)]; %;...
                % avg_dep_var(:,:,:,param_idx,selected_scheme_idx,1:sensor_dimension);...
                % avg_dep_crlb(:,:,:,param_idx,selected_scheme_idx,1:sensor_dimension)];
    if convert_to_log
        all_vals = log10(all_vals);
    end

    for scheme_idx = selected_scheme_idx % 1:length(selected_schemes)
        fig1 = figure;
        set(fig1,'Position',[100,100,675,475])
        ax = gca;

        count_vals = sensor_vals;

        for count_idx = 1:length(count_vals)
            z_vals = squeeze(avg_dep_var(:,:,:,param_idx,scheme_idx,count_idx,1));
            if convert_to_log
                z_vals = log10(z_vals);
            end
            surf(agent_db_values, channel_db_values(1,:,1), z_vals)
        end

        cb = colorbar;
        colormap(turbo)
        % Change view angle to better see both x and y dimensions
        view(150,20)
        xl = xlabel('Channel SNR (dB)', 'FontWeight', 'bold');
        yl = ylabel('Sensor SNR (dB)', 'FontWeight', 'bold');
        
       
        title('$$\hat{' + params_latex(param_idx) + '}$$', 'Interpreter','latex')
        set(ax, 'FontSize', 15);

        zlabel('$$ \mathbf{\log_{10} ( \mathrm{E} \{ (\hat{'+params_latex(param_idx)+'} -' + params_latex(param_idx) + ')^2 \} ) }$$','Interpreter','latex', 'FontSize', 20)
        
        % Add box around plot axes.
        box(ax, 'on');
        grid on
        
        if convert_to_log
            z_lower = 1.05*min(all_vals,[],"all");
            z_upper = 0.95*max(all_vals,[],"all");
        else
            set(ax, 'ZScale', 'log')
            z_lower = 10^(-0.25)*min(all_vals,[],"all");
            z_upper = 1.05*max(all_vals,[],"all");
        end

        zlim([z_lower,z_upper]);
        clim([z_lower,z_upper])

        if ~convert_to_log
            z_ticks = power_10_range(z_lower,z_upper);
            ax.ZTick = z_ticks;
        end

        ax.GridColor = [0 0 0];
        ax.GridAlpha = 1;
        ax.LineWidth = 1;

        xpos = get(ax, 'Xlabel');
        xpos = xpos.Position;
        xpos(1) = xpos(1) + 2;
        xpos(2) = xpos(2) - 2;
        
        xl.Position = xpos;
        xl.Rotation = 10;

        ypos = get(ax, 'Ylabel');
        ypos = ypos.Position;
        ypos(1) = ypos(1) - 2;
        ypos(2) = ypos(2) + 2.5;

        yl.Position = ypos;
        yl.Rotation = -30;

        axis normal

        % Save the figure.
        exportgraphics(fig1, fullfile(save_folder, sprintf(experiment + "_" + string(agent_db_values(end)) + "_db_" + params_text(param_idx) + "_" + selected_schemes(scheme_idx) + ".png")), 'Resolution', 300);
    end
end