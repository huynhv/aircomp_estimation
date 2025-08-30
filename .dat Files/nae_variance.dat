% check for monotonically decreasing variance
seed = 264;
rng(seed);

sensor_vals = 20;

% Set number of deployments.
nDeployments = 10000;
% Set number of trials.
K = 101;

% Set observation window length in seconds.
T_obs = 5;

% Define event signal parameters.
T_s = 1;
w0 = pi;

% Define time vector
dt = T_obs/(K-1);
fs = 1/dt;
t = dt*(0:K-1)';

% Define true parameter values.
alpha_true = 2;
t0_true = 1;

% Calculate the signal energy (such that the signal is assumed to be within the
% interioir fo the observation period).
func = @(t) abs(sensor_signal(t)).^2;
P_s = integral(func,0,T_obs)/T_obs;

% Define range for distribution of mi and ti (tau_i)
max_mi = 1;
min_mi = 0.5;
max_ti = 1;
min_ti = 0;

% Generate channel gains.
all_gi = (randn(1,max(sensor_vals),1,nDeployments) + 1i*randn(1,max(sensor_vals),1,nDeployments))/sqrt(2);

% Generate mi and ti.
all_mi = unifrnd(min_mi,max_mi,1,max(sensor_vals),1,nDeployments);
all_ti = unifrnd(min_ti,max_ti,1,max(sensor_vals),1,nDeployments);

% Define expected value of mi^2.
E_mi_sqr = (1/12) * (max_mi-min_mi)^2 + 0.5*(min_mi+max_mi);

% Define Rayleigh distribution parameters.
rayleigh_factor = 1/sqrt(2);
E_mag_g_sqr = 2*rayleigh_factor^2;


% placeholder constants for noise psd terms (should not make a difference in our
% simulations
agent_db_val = 0;
channel_db_val = 0;

selected_schemes = ["MPC","EPC","PPC","NPC"];


var_results = zeros(1,max(sensor_vals),1,nDeployments,length(selected_schemes));

for sensor_idx = 1:length(sensor_vals)
   
    S = sensor_vals(sensor_idx);
    % Select subset of mi, ti, and gi values.
    mi = all_mi(:,1:S,:,:);
    ti = all_ti(:,1:S,:,:);
    gi = all_gi(:,1:S,:,:);

    % Calculate quantized phase.
    M = 4;
    [qgi,~] = quantize_comp(gi,M);

    % Generate si(t-t0).
    si_true = mi .* sensor_signal(t - ti - t0_true); % K x S
    % Compute Ei.
    Ei = sum(abs(si_true).^2 .* dt,1); % 1 x S
    % Iterate through schemes.
    for scheme_idx = 1:length(selected_schemes)
        scheme = selected_schemes(scheme_idx);
        % Define channel gains and server noise power.
        if scheme == "MPC"
            ci = ones(size(gi));
            gamma_n = dt * P_s * E_mi_sqr / db2magTen(channel_db_val);
        elseif scheme == "EPC"
            ci = abs(gi);
            gamma_n = dt * P_s * E_mi_sqr * E_mag_g_sqr / db2magTen(channel_db_val);
        elseif scheme == "PPC"
            ci = qgi;
            gamma_n = dt * P_s * E_mi_sqr * E_mag_g_sqr / db2magTen(channel_db_val);
        elseif scheme == "NPC"
            ci = gi;
            gamma_n = P_s * E_mi_sqr * E_mag_g_sqr / db2magTen(channel_db_val);
        end

        w_psd_constant = 1;
        gamma_w = (dt/w_psd_constant) * (P_s / db2magTen(agent_db_val));
        scaled_w_psd_constant = gamma_w * w_psd_constant;
        
        n_psd_constant = 1;
        scaled_n_psd_constant = gamma_n * n_psd_constant;
        
        B = 1;

        var_numerator = scaled_w_psd_constant .* cumsum(real(ci).^2 .* Ei) + scaled_n_psd_constant*B;
        var_denominator = cumsum(real(ci) .* Ei).^2;
        var_results(:,:,:,:,scheme_idx) = var_numerator ./ var_denominator;
    end
end

% final dimensions should be S x schemes
avg_var_results = squeeze(mean(var_results, 4));

close all

fig1 = figure;
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact')
set(fig1,'Position',[100,100,600,600])
for i = 1:length(selected_schemes)
    nexttile;
    
    plot(1:max(sensor_vals), (avg_var_results(:,i)), 'LineWidth', 1.5);

    set(gca, 'YScale', 'log');
    
    % Placeholders for titles
    title(selected_schemes(i));   % Change as needed
    xlabel('Number of Sensors','Interpreter','latex');                    % Change as needed
    ylabel('Theoretical Variance','Interpreter','latex');                    % Change as needed
    
    grid on;                                   % Optional: show grid
    hold on;                                   % Optional if adding more plots
end

% Add a super title for the whole figure
% sgtitle(' For Increasing Number of Sensors');   % Replace with your desired title
sgtitle('Theoretical Variance of $$\hat{\alpha}$$ vs. Number of Sensors', 'Interpreter','latex')
save_folder = "C:\Users\Vincent Huynh\Desktop\Final AirComp Results\monotonic_variance";
if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end
exportgraphics(fig1, fullfile(save_folder, "nae_variance.png"), 'Resolution', 300);


%% functions
% QUANTIZE_COMP Returns the quantized sensor chanel phase.
function [qg,cq] = quantize_comp(g,M)
    phi = angle(g);
    phi = phi + (2*pi)*(phi<0);
    
    all_cq = 0;

    for l = 1:M
        if l == 1
            cq_temp = exp(-(l-1)*1i*2*pi/M) * ((phi > (2*M-1)*pi/M) | (phi <= pi/M));
        else
            cq_temp = exp(-(l-1)*1i*2*pi/M) * ((phi > (2*(l-1)-1)*pi/M) & (phi <= (2*l-1)*pi/M));
        end
        all_cq = all_cq + cq_temp;
    end
    qg = g .* all_cq;
    cq = all_cq;
end