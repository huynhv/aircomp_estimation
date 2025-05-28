%% use this one to generate paper figures
% 5-23-2025: fixed dt scaling for NAE, derived closed-form for variance
% 5-27-2025: fixed dt scaling for CWE
%% assume channel is unknown at the receiver, iterate over deployments
clear all
close all

% top seed: 264, dropout seed: 22, 100 deps 500 trials

% set a seed for reproducibility
seed = 264;
rng(seed);

%%% EXPERIMENT CONTROLS %%%
% naive_joint, naive_disjoint, joint, disjoint, agent_server_noise, random_dropout
experiment = "joint";
%%%

% number of sensors
% if experiment == "random_dropout"
%     sensor_vals = [10,8,6];
% else
%     sensor_vals = [1,5,10];
% end
sensor_vals = [5,10];

% define sensor snr
agent_db_values = [-10];

if experiment == "random_dropout"
    dropout_vals = [0,2,4];
    sensor_dimension = length(dropout_vals);
else
    dropout_vals = 0;
    sensor_dimension = length(sensor_vals);
end

% number of deployments
nDeployments = 50; % 150
% number of trials
nTrials = 500; % 500
% number of measurements per sensor, starting from 0 then 100 sample means K = 101
K = 101; % 101

% Window length in seconds
T_obs = 5;
% period of our expected signal
T_s = 1;
w0 = pi;

% time divisions
dt = T_obs/(K-1);
% time vector
t = dt*(0:K-1)';

% calculate the signal energy on the interior of the observation period
func = @(t) abs(sensor_signal(t)).^2;
P_s = integral(func,0,T_obs);

% true parameter values
alpha_true = 2;
t0_true = 1;

% range for distribution of m
upper_m = 1;
lower_m = 0.5;
max_tau = 1;

% maximum t0 value to guarantee the signal is in the interior of the
% observation period
t0_max = T_obs-T_s-max_tau;

% noise dimensions: K x S x Trials x Deployments

% when we sample AWGN, have to divide simulated variance by dt because in
% reality we are applying an anti-aliasing filter before sampling
all_w = randn(K,max(sensor_vals),nTrials,nDeployments);
% first two columns used for iterative algorithm, last one used for y
n = (randn(K,1,nTrials,nDeployments) + 1i*randn(K,1,nTrials,nDeployments));

% generate channel gains
all_g = (randn(1,max(sensor_vals),1,nDeployments) + 1i*randn(1,max(sensor_vals),1,nDeployments))/sqrt(2);

% Rayleigh scale parameter
rayleigh_factor = 1/sqrt(2);
E_mag_g_sqr = 2*rayleigh_factor^2;

all_schemes = ["MPC","EPC","PPC"];
selected_schemes = all_schemes;

% generate m and tau
all_m = unifrnd(lower_m,upper_m,1,max(sensor_vals),1,nDeployments);
all_tau = unifrnd(0,max_tau,1,max(sensor_vals),1,nDeployments);

E_mi_sqr = (1/12) * (upper_m-lower_m)^2 + 0.5*(lower_m+upper_m);

% define channel snr
% we're modeling an analog quantity in a digital domain, if we set channel SNR to infinity, then of
% course we're going to get results that don't make sense...the variance on t0 estimator would be 0
% (since we're sampling digitially)

% we need to be mindful of the practical gaps that exist between analytical expressions and modeling
% them in using MATLAB
channel_db_values = repmat([-10,-5,0,5,10], length(selected_schemes), 1, length(agent_db_values));

% initialize empty matrices for crlb, variance, mse, and bias
% alpha is index 1, t0 is index 2
crlb = zeros(1,length(agent_db_values),size(channel_db_values,2),2,length(selected_schemes),sensor_dimension,nDeployments);
empirical_var = crlb;
empirical_mse = crlb;
bias = crlb;
my_var = crlb;

% start run timer
loopTic = tic;

% iterate through sensor snr values
for agent_db_idx = 1:length(agent_db_values)
    disp('')
    disp("**Agent SNR = " + (agent_db_values(agent_db_idx)) + " dB**")
    
    agent_db = agent_db_values(agent_db_idx);
    w_psd_constant = 1;
    gamma_w = (dt/w_psd_constant) * (P_s / db2magTen(agent_db));
    
    scaled_w = sqrt(gamma_w * w_psd_constant /dt) * all_w; % --> variance of this should be gamma_w/dt
    scaled_w_psd_constant = gamma_w * w_psd_constant;
    % w_psd_function_td = gamma_w * (w_psd_constant/dt);

    if experiment == "agent_server_noise"
        ratio_arr = [0.25,0.5,1,2,4];
        for scheme_idx = 1:length(selected_schemes)
            scheme = selected_schemes(scheme_idx);
            if scheme == "MPC"
                channel_db_values(scheme_idx,:,agent_db_idx) = magTen2db(P_s .* E_mi_sqr .* ratio_arr ./ scaled_w_psd_constant);
            elseif scheme == "EPC"
                channel_db_values(scheme_idx,:,agent_db_idx) = magTen2db(P_s .* E_mi_sqr .* E_mag_g_sqr .* ratio_arr ./ scaled_w_psd_constant);
            elseif scheme == "PPC"
                channel_db_values(scheme_idx,:,agent_db_idx) = magTen2db(P_s .* E_mi_sqr .* E_mag_g_sqr .* ratio_arr ./ scaled_w_psd_constant);
            elseif scheme == "NPC"
                channel_db_values(scheme_idx,:,agent_db_idx) = magTen2db(P_s .* E_mi_sqr .* E_mag_g_sqr .* ratio_arr ./ scaled_w_psd_constant);
            end
         end
    end

    % iterate through sensor values
    for sensor_idx = 1:length(sensor_vals)
        S = sensor_vals(sensor_idx);

        if (experiment == "random_dropout" && S ~= max(sensor_vals))
            continue
        end

        if experiment == "random_dropout"
            % 151, 20, 90, 110
            seed = 151;
            rng(seed);
            which_dropout = zeros(nDeployments,max(dropout_vals));
            for idx = 1:nDeployments
                which_dropout(idx,:) = randperm(max(sensor_vals),max(dropout_vals));
            end
        end
        
        for dropout_idx = 1:length(dropout_vals)
            num_dropout = dropout_vals(dropout_idx);
            % select subset of m, tau, and g values
            m_sensors = all_m(:,1:S,:,:);
            tau_sensors = all_tau(:,1:S,:,:);

            % tau_sensors = zeros(size(m_sensors));
            g = all_g(:,1:S,:,:);
    
            % calculate quantized phase compensation
            M = 4;
            [qg,qg_cq] = quantize_comp(g,M);
           
            s_i_true = m_sensors .* sensor_signal(t - tau_sensors - t0_true); % K x S
            x_i = alpha_true * s_i_true + scaled_w(:,1:S,:,:); % K x S x nTrials x nDeployments
    
            % iterate through channel snr values
            for channel_db_idx = 1:size(channel_db_values,2)
                for scheme_idx = 1:length(selected_schemes)
                    scheme = selected_schemes(scheme_idx);
                    
                    disp("S = " + S)
                    disp("Dropout = " + dropout_vals(dropout_idx))
                    disp(scheme + ", Channel SNR = " + (channel_db_values(scheme_idx,channel_db_idx,agent_db_idx)) + " dB")
                    disp(" ")
    
                    if scheme == "MPC"
                        channel_gains = ones(size(g));
                        gamma_n = dt * P_s * E_mi_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                    elseif scheme == "EPC"
                        channel_gains = abs(g);
                        gamma_n = dt * P_s * E_mi_sqr * E_mag_g_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                    elseif scheme == "PPC"
                        channel_gains = qg;
                        gamma_n = dt * P_s * E_mi_sqr * E_mag_g_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                    end
                    % elseif scheme == "NPC"
                    %     channel_gains = g;
                    %     gamma_n = P_s * E_mi_sqr * E_mag_g_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                    % end

                    if num_dropout > 0
                        dropout_elements = which_dropout(:,1:num_dropout);
                        for item = 1:nDeployments
                            channel_gains(1,dropout_elements(item,:),1,item) = 0;
                            m_sensors(1,dropout_elements(item,:),1,item) = 0;
                        end
                    end
    
                    n_psd_constant = 1;
                    scaled_n = sqrt( (gamma_n*n_psd_constant/dt) / 2 ) * n;
                    % psd function of n in time domain (td)
                    n_psd_function_td = gamma_n/dt;
                    scaled_n_psd_constant = gamma_n * n_psd_constant;

                    %%%%%%%%%%%%%%%%% ESTIMATION %%%%%%%%%%%%%%%%%
   
                    % find peak index from noisy channel-weighted sum (cws)
                    cws = reshape(matched_filter_integral(reshape(x_i,K,S,1,nTrials,nDeployments), reshape(m_sensors .* sensor_signal(t-tau_sensors-reshape(t,1,1,length(t))), K,S,K,1,nDeployments), channel_gains, K, S, nTrials, nDeployments, dt),1,K,nTrials,nDeployments);
                    y =  cws + pagetranspose(scaled_n);

                    % we know a priori that the estimate of t0 cannot exceed t0_max
                    if experiment == "naive_joint" || experiment == "naive_disjoint"
                        [~,I] = max(real(y(:,1:(t0_max/dt + 1),:,:)));
                    else
                        [~,I] = max(abs(y(:,1:(t0_max/dt + 1),:,:)));
                    end
                    t0_estimates_for_plot = (I-1)*dt;
                    
                    if experiment == "disjoint"
                        t0_estimates = t0_true*ones(1,1,nTrials,nDeployments);
                    else
                        t0_estimates = t0_estimates_for_plot;
                    end

                    exp_split = strsplit(experiment,'_');
                    
                    if exp_split(1) == "naive"
                        % Assume y is 1x101x500x100
                        % Assume I is 1x1x500x100, containing indices for the 2nd dimension
                        
                        % Reshape I to match the size of the 3rd and 4th dimensions of y
                        if exp_split(2) == "joint"
                            II = squeeze(I); % I becomes 500x100
                        elseif exp_split(2) == "disjoint"
                            II = (t0_true / dt + 1) * ones(nTrials,nDeployments);
                        end

                        % Generate the subscript indices for the 3rd and 4th dimensions
                        [dim3, dim4] = ndgrid(1:size(y, 3), 1:size(y, 4));
                        
                        % Create linear indices for the desired elements
                        linear_indices = sub2ind(size(y), ones(size(II)), II, dim3, dim4);
                        
                        % Extract the elements from y
                        peak_y_vals = y(linear_indices);
                        
                        % Result is the specific element from each column
                        peak_y_vals = reshape(peak_y_vals,size(I));

                        E_i = sum(abs(s_i_true).^2 .* dt,1); % 1 x S

                        alpha_estimates =  real(peak_y_vals) ./ sum(real(channel_gains) .* E_i,2);
                    end
    
                    if exp_split(1) ~= "naive"
                        fs = 1/dt;
                        L = K-1;
                        f = 0:fs/(2*L):fs/2;
                        indet_idx = find(abs(round(f,4)) == 0.5);
                        omega = 2*pi*f;
    
