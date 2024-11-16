%% assume channel is unknown at the receiver, iterate over deployments
% 3/29/2024 - apply revisions per Prof. Ding's notes
% 4/8/2024 - loop over sensor confidence values
% 4/19/2024 - Weiwei reviewed the code with Vincent
% 4/29/2024 - discuss with Weiwei and change approach to h
% 5/6/2024 - sampling the noise means scaling by 1/dt because of anti-alias LPF
% 5/7/2024 - slim down the code
% 5/8/2024 - v8 start going over trials per deployment
% 6/12/2024 - v9 implement decorrelation and whitening approach, also
% change PPC to use the real and imaginary parts of the signal
% 6/18/2024 - v10 modify Omega to use matrix multiplication
% 6/28/2024 - v12 (skip v11 not to confuse Weiwei) add matched filter matrix multiplication function
% 7/5/2024 - v13 fix FFT for Qn, hdI, and SQnR/SQnI, address hdI being infinity
% 7/19/2024 - add MLE
% 7/23/2024 - go line by line to verify
% 7/31/2024 - Update hdI to be a delta function
% 8/9/2024 - go back to hdI = 1/(gamma + SEs)
% 8/14/2024 - added to git repo
clear all
close all

% set a seed for reproducibility
seed = 264;
rng(seed);

%%% EXPERIMENT CONTROLS %%%
% Paper 1
% joint, disjoint, agent_server_noise,  dropout_participation, random_dropout
experiment = "joint";
%%%

% number of sensors
sensor_vals = [1,5,10];

% define sensor snr
agent_db_values = [20];

if experiment == "random_dropout"
    dropout_vals = [0,2,4];
    sensor_dimension = length(dropout_vals);
else
    dropout_vals = 0;
    sensor_dimension = length(sensor_vals);
end

% number of deployments
nDeployments = 100;
% number of trials
nTrials =  500;
% number of measurements per sensor, starting from 0 then 100 sample means K = 101
K = 101;

% Window length in seconds
T_obs = 5;
% period of our expected signal
T_s = 1;
w0 = pi;

% ---------------------------------- %
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
all_w = randn(K,max(sensor_vals),nTrials,nDeployments)/sqrt(dt);
% first two columns used for iterative algorithm, last one used for y
n = (randn(K,3,nTrials,nDeployments) + 1i*randn(K,3,nTrials,nDeployments))/sqrt(dt);

% generate channel gains
all_g = (randn(1,max(sensor_vals),1,nDeployments) + 1i*randn(1,max(sensor_vals),1,nDeployments))/sqrt(2);

% Rayleigh scale parameter
rayleigh_factor = 1/sqrt(2);
E_mag_g_sqr = 2*rayleigh_factor^2;

all_schemes = ["EPC","PPC","NPC"];

selected_schemes = all_schemes;

save_t0 = [];
save_alpha = [];

% generate m and tau
all_m = unifrnd(lower_m,upper_m,1,max(sensor_vals));
% E_mi_sqr = (1/12) * (upper_m-lower_m)^2 + 0.5*(lower_m+upper_m);
E_mi_sqr = mean(all_m.^2);

all_tau = unifrnd(0,max_tau,1,max(sensor_vals));

% define channel snr
channel_db_values = repmat([10,15,20,25,30], length(selected_schemes), 1, length(agent_db_values));

% initialize empty matrices for crlb, variance, mse, and bias
% alpha is index 1, t0 is index 2
crlb = zeros(1,length(agent_db_values),size(channel_db_values,2),2,length(selected_schemes),sensor_dimension,nDeployments);
empirical_var = crlb;
empirical_mse = crlb;
bias = crlb;

% start run timer
loopTic = tic;

% iterate through sensor snr values
for agent_db_idx = 1:length(agent_db_values)
    disp('')
    disp("**Agent SNR = " + (agent_db_values(agent_db_idx)) + " dB**")
    agent_db = agent_db_values(agent_db_idx);    
    var_w = P_s / db2magTen(agent_db);
    scaled_w = sqrt(var_w) * all_w;

    if experiment == "agent_server_noise"
        ratio_arr = [0.25,0.5,1,2,4];
        for scheme_idx = 1:length(selected_schemes)
            scheme = selected_schemes(scheme_idx);
            if scheme == "MLE"
                continue
            elseif scheme == "EMPC"
                channel_db_values(scheme_idx,:,agent_db_idx) = magTen2db(P_s .* E_mi_sqr .* ratio_arr ./ var_w);
            elseif scheme == "EPC"
                channel_db_values(scheme_idx,:,agent_db_idx) = magTen2db(P_s .* E_mi_sqr .* E_mag_g_sqr .* ratio_arr ./ var_w);
            elseif scheme == "PPC"
                channel_db_values(scheme_idx,:,agent_db_idx) = magTen2db(P_s .* E_mi_sqr .* E_mag_g_sqr .* ratio_arr ./ var_w);
            elseif scheme == "NPC"
                channel_db_values(scheme_idx,:,agent_db_idx) = magTen2db(P_s .* E_mi_sqr .* E_mag_g_sqr .* ratio_arr ./ var_w);
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
            seed = 22;
            rng(seed);
            which_dropout = zeros(nDeployments,max(dropout_vals));
            for idx = 1:nDeployments
                which_dropout(idx,:) = randperm(max(sensor_vals),max(dropout_vals));
            end
        end
        
        for dropout_idx = 1:length(dropout_vals)
            % select subset of m, tau, and g values
            m_sensors = all_m(:,1:S);
            tau_sensors = all_tau(:,1:S);
            g = all_g(:,1:S,:,:);
    
            % calculate quantized phase compensation
            M = 4;
            [qg,qg_cq] = quantize_comp(g,M);
           
            s_i_true = m_sensors .* sensor_signal(t - tau_sensors - t0_true); % K x S
            x_i = alpha_true * s_i_true + scaled_w(:,1:S,:,:); % K x S x nTrials x nDeployments

            % iterate through channel snr values
            for channel_db_idx = 1:length(channel_db_values)
                for scheme_idx = 1:length(selected_schemes)
                    scheme = selected_schemes(scheme_idx);
                    disp("S = " + S)
                    disp("Dropout = " + dropout_vals(dropout_idx))
                    disp(scheme + ", Channel SNR = " + (channel_db_values(scheme_idx,channel_db_idx,agent_db_idx)) + " dB")
    
                    if scheme == "MLE"
                        if channel_db_idx > 1
                            continue
                        else
                            channel_gains = ones(size(g));
                            E_i = repmat(sum(abs(s_i_true).^2 .* dt,1),1,1,1,nDeployments); % 1 x S
                            var_n = 0;
                        end
                    elseif scheme == "EMPC"
                        channel_gains = ones(size(g));
                        var_n = P_s * E_mi_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                    elseif scheme == "EPC"
                        channel_gains = abs(g);
                        var_n = P_s * E_mi_sqr * E_mag_g_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                    elseif scheme == "PPC"
                        channel_gains = qg;
                        var_n = P_s * E_mi_sqr * E_mag_g_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                    elseif scheme == "NPC"
                        channel_gains = g;
                        var_n = P_s * E_mi_sqr * E_mag_g_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                    end

                    num_dropout = dropout_vals(dropout_idx);
                    if num_dropout > 0
                        dropout_elements = which_dropout(:,1:num_dropout);
                        for item = 1:nDeployments
                            channel_gains(1,dropout_elements(item,:),1,item) = 0;
                        end

                    end
    
                    scaled_n = sqrt(var_n/2) * n;
                    %%%%%%%%%%%%%%%%% ESTIMATION %%%%%%%%%%%%%%%%%
    
                    % perform golden-section search over t0
                    % left_val = zeros(1,1,nTrials,nDeployments);
                    % right_val = t0_max*ones(1,1,nTrials,nDeployments);
                    % j = 0;
                    % 
                    % while (true) 
                    %     d = (right_val-left_val)/GR;
                    %     x1 = left_val + d;
                    %     x2 = right_val - d;
                    % 
                    %     if (sum(sum(abs(x1-x2) < 1E-3)) == nTrials*nDeployments)
                    %         break;
                    %     end
                    %     j = j + 1;
                    % 
                    %     channel_noise_1 = scaled_n(j,1,:,:);
                    %     channel_noise_2 = scaled_n(j,2,:,:);
                    % 
                    %     % there are two different time indices because the noise should be independent
                    %     fx1 = calculate_y(x1,x_i,m_sensors,t-tau_sensors,channel_noise_1,dt,channel_gains);
                    %     fx2 = calculate_y(x2,x_i,m_sensors,t-tau_sensors,channel_noise_2,dt,channel_gains);
                    % 
                    %     % compare magnitudes to find swap indices
                    %     swap_lv = abs(fx1) > abs(fx2);
                    %     swap_rv = ~swap_lv;
                    % 
                    %     left_val(swap_lv) = x2(swap_lv);
                    %     right_val(swap_rv) = x1(swap_rv);
                    % end
                    % t0_estimates_for_plot = (x1+x2)/2;

                    % each time index y(t) requires its own matrix
                    % multiplication hence I added the singleton dimension
                    cws = reshape(matched_filter_integral(reshape(x_i,K,S,1,nTrials,nDeployments), m_sensors .* sensor_signal(t-tau_sensors-reshape(t,1,1,length(t))), channel_gains, K, S, nTrials, nDeployments, dt),1,K,nTrials,nDeployments);
                    y =  cws + pagetranspose(scaled_n(:,1,:,:));

                    % we know a priori that the estimate of t0 cannot
                    % exceed t0_max
                    [~,I] = max(abs(y(:,1:(t0_max/dt + 1),:,:)));
                    t0_estimates_for_plot = (I-1)*dt;
                    disp("")
                    
                    if experiment == "disjoint"
                        t0_estimates = t0_true*ones(size(x1));
                    else
                        t0_estimates = t0_estimates_for_plot;
                        % t0_estimates = t0_true*ones(size(x1));
                    end
                    
                    if scheme == "MLE"
                        alpha_estimates = sum(channel_gains .* sum(x_i .* m_sensors .* sensor_signal(t-tau_sensors-t0_estimates),1) * dt,2) ./ sum(E_i,2);
                    end
    
                    if scheme ~= "MLE"

                        fs = 1/dt;
                        L = K-1;
                        f = 0:fs/(2*L):fs/2;
                        indet_idx = find(abs(round(f,4)) == 0.5);
                        omega = 2*pi*f;
    

                        if (scheme == "NPC" || scheme == "PPC")
                            real_channel = real(channel_gains);
                            imag_channel = imag(channel_gains);
        
                            b = sum(real_channel.*imag_channel.*m_sensors.^2,2) ./ (sum(m_sensors.^2 .* (real_channel.^2 - imag_channel.^2),2));
    
                            disc = 1 + 4.*b.^2;
                            a1 = (-1 + sqrt(disc)) ./ (2*b);
                            a2 = (-1 - sqrt(disc)) ./ (2*b);
                            a = a1;
    
                            SEs = ones(1,K,1,nDeployments) .* (2 * pi^2 * (1+cos(omega)) ./ (omega.^2 - pi^2).^2);
                            gamma = 1;
    
                            mag_sqr_SHdI = 1 ./ (gamma + SEs);
    
                            for idx=indet_idx
                                disp('filling indeterminate values')
                                % fill by L'Hopital's
                                mag_sqr_SHdI(1,idx,:) = 8*w0^2 / (gamma*8*w0^2 - 2*pi^2*cos(w0));
                            end
    
                            mag_SHdI = sqrt(mag_sqr_SHdI);
    
                            [hdI, hdI_matrix] = get_time_domain(mag_SHdI,dt,nDeployments);
        
                            hdR_matrix = reshape(a,1,1,nDeployments) .* hdI_matrix;
    
                            hd_matrix = hdR_matrix + 1i*hdI_matrix;
    
                            pR = var_w * sum(m_sensors.^2 .* (a.*real_channel - imag_channel).^2,2);
                            qR = (var_n/2 * (a.^2 + 1)) - gamma*pR;
    
                            pI = var_w * sum(m_sensors.^2 .* (real_channel + a.*imag_channel).^2,2);
                            qI = (var_n/2 * (a.^2 + 1)) - gamma*pI;
    
                            SKndR = pR + mag_sqr_SHdI .* qR;
                            SKndI = pI + mag_sqr_SHdI .* qI;
    
                            if sum(SKndR < 0, "all")
                                error("SKndR less than 0!")
                            elseif sum(SKndI < 0, "all")
                                error("SKndI less than 0!")
                            end
    
                            SQnR = 1 ./ SKndR;
                            SQnI = 1 ./ SKndI;
        
                            [~,QnR_matrix] = get_time_domain(SQnR,dt,nDeployments);
                            [~,QnI_matrix] = get_time_domain(SQnI,dt,nDeployments);
    
                            %%% for alpha
                            % I didn't know what to name this but it's sum gi int si(tau - t0_estimates) si(tau-t) dtau
                            % SINCE EACH ESTIMATED t0 is DIFFERENT, then WE HAVE DIFFERENT VALUES ACROSS TRIALS
                            aaa = reshape(matched_filter_integral(m_sensors .* reshape(sensor_signal(t-tau_sensors-t0_estimates),K,S,1,nTrials,nDeployments), m_sensors .* sensor_signal(t-tau_sensors-reshape(t,1,1,length(t))), channel_gains, K, S, nTrials, nDeployments, dt),1,K,nTrials,nDeployments);
    
                            aaaR = real(aaa);
                            aaaI = imag(aaa);
    
                            % the d here stands for decorrelation, NOT derivative,
                            % matches paper notation exactly
                            aaadR = dt*(pagemtimes(aaaR,reshape(hdR_matrix,K,K,1,nDeployments)) - pagemtimes(aaaI,reshape(hdI_matrix,K,K,1,nDeployments)));
                            aaadI = dt*(pagemtimes(aaaI,reshape(hdR_matrix,K,K,1,nDeployments)) + pagemtimes(aaaR,reshape(hdI_matrix,K,K,1,nDeployments)));
    
                            yR = real(y);
                            yI = imag(y);
        
                            ydR = dt*(pagemtimes(yR,reshape(hdR_matrix,K,K,1,nDeployments)) - pagemtimes(yI,reshape(hdI_matrix,K,K,1,nDeployments)));
                            ydI = dt*(pagemtimes(yI,reshape(hdR_matrix,K,K,1,nDeployments)) + pagemtimes(yR,reshape(hdI_matrix,K,K,1,nDeployments)));
    
                            resh_QnR_matrix = reshape(QnR_matrix,length(t),length(t),1,nDeployments);
                            resh_QnI_matrix = reshape(QnI_matrix,length(t),length(t),1,nDeployments);
    
                            num1 = pagemtimes(pagemtimes(ydR,resh_QnR_matrix),pagetranspose(aaadR));
                            num2 = pagemtimes(pagemtimes(ydI,resh_QnI_matrix),pagetranspose(aaadI));
                            num = dt*dt*(num1+num2);
    
                            denom1 = pagemtimes(pagemtimes(aaadR,resh_QnR_matrix),pagetranspose(aaadR));
                            denom2 = pagemtimes(pagemtimes(aaadI,resh_QnI_matrix),pagetranspose(aaadI));
                            denom = dt*dt*(denom1+denom2);
                            %%%
                        else
                            Beta = 2 * pi^2 * sum(channel_gains.^2 .* m_sensors.^2,2);
                            C1 = Beta .* var_w;
                            C2 = var_n/2;
    
                            V = Beta .* var_w .*(1+cos(omega)) ./ (omega.^2 - pi^2).^2 + var_n/2;
                                
                            mag_sqr_H = 1./V;
                
                            for idx=indet_idx
                                % fill by L'Hopital's
                                disp('filling indeterminate values')
                                % mag_sqr_H(1,idx,:) = 8*pi^2 / (Beta .* var_w); 
                                mag_sqr_H(1,idx,:) = 8*pi^2 / (C2*8*pi^2 + C1);
                            end
    
                            [~,Qn_matrix] = get_time_domain(mag_sqr_H,dt,nDeployments);
                            Omega = matched_filter_integral(m_sensors .* reshape(sensor_signal(t-tau_sensors-t0_estimates),K,S,1,nTrials,nDeployments), m_sensors .* sensor_signal(t-tau_sensors-reshape(t,1,1,length(t))), channel_gains, K, S, nTrials, nDeployments, dt);
                        
                            num = dt*dt*pagemtimes(pagemtimes(real(y),reshape(Qn_matrix,length(t),length(t),1,nDeployments)),pagetranspose(reshape(Omega,1,length(t),nTrials,nDeployments)));
                            denom = dt*dt*pagemtimes(pagemtimes(reshape(Omega,1,length(t),nTrials,nDeployments),reshape(Qn_matrix,length(t),length(t),1,nDeployments)),pagetranspose(reshape(Omega,1,length(t),nTrials,nDeployments)));
                        end
                        % we determine the magnitude of alpha
                        alpha_estimates = num ./ denom;
                    end

                    if experiment == "random_dropout"
                        save_dim = dropout_idx;
                    else
                        save_dim = sensor_idx;
                    end
    
                    empirical_var(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:) = var(alpha_estimates,0,3);
                    empirical_var(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:) = var(t0_estimates_for_plot,0,3);
    
                    empirical_mse(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:) = mean((alpha_estimates - alpha_true).^2,3);
                    empirical_mse(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:) = mean((t0_estimates_for_plot - t0_true).^2,3);
    
                    bias(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:) = mean(alpha_estimates,3);
                    if any(isnan(mean(alpha_estimates,3)))
                        disp("")
                    end
                    bias(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:) = mean(t0_estimates_for_plot,3);
    
                    %%%%%%%%%%%%%%%%% CRLB %%%%%%%%%%%%%%%%%
    
                    % Calculate the CRLBs
                    dsalpha = reshape(squeeze(sum(channel_gains .* sum(m_sensors .* sensor_signal(t-tau_sensors-t0_true) .* m_sensors .* sensor_signal(t-tau_sensors-reshape(t,1,1,length(t))) * dt,1),2)),K,1,nDeployments);
                    temp = -w0*cos(w0*(t-tau_sensors-t0_true)).*(heaviside(t-tau_sensors-t0_true) - heaviside(t-tau_sensors-t0_true-T_s)); % + ...
                        % sin(w0*(t-tau_sensors-t0_true)).*-1.*(dirac(t-tau_sensors-t0_true) - dirac(t-tau_sensors-t0_true-T_s));
                    dst0 = reshape(squeeze(sum(channel_gains .* sum(alpha_true .* m_sensors.^2 .* temp .* sensor_signal(t-tau_sensors-reshape(t,1,1,length(t))) * dt,1),2)),K,1,nDeployments);
            
                    if scheme == "MLE"
                        crlb(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:) = var_w / sum(E_i,2);
                        temp = -w0*cos(w0*(t-tau_sensors-t0_true)).*(heaviside(t-tau_sensors-t0_true) - heaviside(t-tau_sensors-t0_true-T_s)) + ...
                        sin(w0*(t-tau_sensors-t0_true)).*-1.*(dirac(t-tau_sensors-t0_true) - dirac(t-tau_sensors-t0_true-T_s));
                        crlb(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:) = var_w / sum(sum((alpha_true .* m_sensors .* temp).^2 * dt,1),2);
                    elseif (scheme == "NPC" || scheme == "PPC")
                            
                        %%% for alpha
                        dsdR_alpha = dt*(pagemtimes(pagetranspose(real(dsalpha)),hdR_matrix) - pagemtimes(pagetranspose(imag(dsalpha)),hdI_matrix));
                        dsdI_alpha = dt*(pagemtimes(pagetranspose(imag(dsalpha)),hdR_matrix) + pagemtimes(pagetranspose(real(dsalpha)),hdI_matrix));

                        % temp_alpha = dt*dt*(pagemtimes(pagemtimes(dsdR_alpha,QnR_matrix),pagetranspose(dsdR_alpha)) + pagemtimes(pagemtimes(dsdI_alpha,QnI_matrix),pagetranspose(dsdI_alpha)));
                        
                        %%% for t0
                        dsdR_t0 = dt*(pagemtimes(pagetranspose(real(dst0)),hdR_matrix) - pagemtimes(pagetranspose(imag(dst0)),hdI_matrix));
                        dsdI_t0 = dt*(pagemtimes(pagetranspose(imag(dst0)),hdR_matrix) + pagemtimes(pagetranspose(real(dst0)),hdI_matrix));

                        % temp_t0 = dt*dt*(pagemtimes(pagemtimes(dsdR_t0,QnR_matrix),pagetranspose(dsdR_t0)) + pagemtimes(pagemtimes(dsdI_t0,QnI_matrix),pagetranspose(dsdI_t0)));
                     
                        fisher_matrix = dt*dt*[(pagemtimes(pagemtimes(dsdR_alpha,QnR_matrix),pagetranspose(dsdR_alpha)) + pagemtimes(pagemtimes(dsdI_alpha,QnI_matrix),pagetranspose(dsdI_alpha)))...
                                    (pagemtimes(pagemtimes(dsdR_alpha,QnR_matrix),pagetranspose(dsdR_t0)) + pagemtimes(pagemtimes(dsdI_alpha,QnI_matrix),pagetranspose(dsdI_t0)));...
                                    (pagemtimes(pagemtimes(dsdR_alpha,QnR_matrix),pagetranspose(dsdR_t0)) + pagemtimes(pagemtimes(dsdI_alpha,QnI_matrix),pagetranspose(dsdI_t0)))...
                                    (pagemtimes(pagemtimes(dsdR_t0,QnR_matrix),pagetranspose(dsdR_t0)) + pagemtimes(pagemtimes(dsdI_t0,QnI_matrix),pagetranspose(dsdI_t0)))];

                        inv_zz = pageinv(fisher_matrix);
                        crlb(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:) = inv_zz(1,1,:);
                        crlb(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:) = inv_zz(2,2,:);
                    else
                        fisher_matrix = dt*dt*[pagemtimes(pagemtimes(pagetranspose(dsalpha),Qn_matrix),dsalpha), pagemtimes(pagemtimes(pagetranspose(dst0),Qn_matrix),dsalpha);...
                            pagemtimes(pagemtimes(pagetranspose(dst0),Qn_matrix),dsalpha), pagemtimes(pagemtimes(pagetranspose(dst0),Qn_matrix),dst0)];
                        inv_zz = pageinv(fisher_matrix);
                        crlb(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:) = inv_zz(1,1,:);
                        crlb(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:) = inv_zz(2,2,:);
                    end
                end
                disp('')
            % figure
            % histogram(alpha_estimates)
            end
        end
    end
end

runtime = toc(loopTic);
disp(floor(runtime/60) + " minutes " + mod(runtime,60) + " seconds")

%% save results
% if experiment == "joint"
%     save("joint_results.mat")
% elseif experiment == "disjoint"
%     save("disjoint_results.mat")
% elseif experiment == "agent_server_noise"
%     save("agent_server_ratio.mat")
% elseif experiment == "random_dropout"
%     save("random_dropout.mat")
% elseif experiment == "dropout_participation"
%     save("dropout_participation.mat")
% else
%     disp("Unknown experiment specified. Results not saved!")
% end

%% avg linear plots
avg_dep_var = mean(empirical_var,length(size(empirical_var)));
avg_dep_mse = mean(empirical_mse,length(size(empirical_mse)));
avg_dep_crlb = mean(crlb,length(size(crlb)));

params = ["\alpha","t_0"];
% color_vec = ["#A2142F","#0072BD","#7E2F8E"];
color_vec = ["#0072BD", "#D95319", "#77AC30"];

for agent_db_idx = 1:length(agent_db_values)
    fig1 = figure;
    set(fig1,'WindowState', 'maximized');
    % set(fig1,'Position',[100,100,1600,400])
    tiledlayout(2,4);
    % don't plot the MLE, use it as baseline curve
    for scheme_idx = find(~strcmp(selected_schemes,"MLE")) 
        % for alpha and t0
        for jj = 1:2
            all_vals = [];
            nexttile
            hold on

            if experiment == "random_dropout"
                for dropout_idx = 1:length(dropout_vals)
                    plot(nan,nan,'color',color_vec(dropout_idx));
                end
            else
                for sensor_idx = 1:length(sensor_vals)
                    plot(nan,nan,'color',color_vec(sensor_idx));
                end
            end
            plot(nan,nan,'^','color','black');
            plot(nan,nan,'x','color','black');
            plot(nan,nan,'o','color','black');

            if experiment == "random_dropout"
                for dropout_idx = 1:length(dropout_vals)
                    plot(channel_db_values(scheme_idx,:,agent_db_idx),(squeeze(avg_dep_var(:,agent_db_idx,:,jj,scheme_idx,dropout_idx,1))),'--^','color',color_vec(dropout_idx))
                    plot(channel_db_values(scheme_idx,:,agent_db_idx),(squeeze(avg_dep_mse(:,agent_db_idx,:,jj,scheme_idx,dropout_idx,1))),'--x','color',color_vec(dropout_idx))
                    plot(channel_db_values(scheme_idx,:,agent_db_idx),(squeeze(avg_dep_crlb(:,agent_db_idx,:,jj,scheme_idx,dropout_idx,1))),'--o','color',color_vec(dropout_idx))
                    all_vals = [all_vals; squeeze(avg_dep_var(:,agent_db_idx,:,jj,scheme_idx,dropout_idx,1));...
                    squeeze(avg_dep_mse(:,agent_db_idx,:,jj,scheme_idx,dropout_idx,1));...
                    squeeze(avg_dep_crlb(:,agent_db_idx,:,jj,scheme_idx,dropout_idx,1))];
                end
            else
                for sensor_idx = 1:length(sensor_vals)
                    if experiment == "agent_server_noise"
                        plot(ratio_arr,(squeeze(avg_dep_var(:,agent_db_idx,:,jj,scheme_idx,sensor_idx,1))),'--^','color',color_vec(sensor_idx))
                        plot(ratio_arr,(squeeze(avg_dep_mse(:,agent_db_idx,:,jj,scheme_idx,sensor_idx,1))),'--x','color',color_vec(sensor_idx))
                        plot(ratio_arr,(squeeze(avg_dep_crlb(:,agent_db_idx,:,jj,scheme_idx,sensor_idx,1))),'--o','color',color_vec(sensor_idx))
                    else
                        plot(channel_db_values(scheme_idx,:,agent_db_idx),(squeeze(avg_dep_var(:,agent_db_idx,:,jj,scheme_idx,sensor_idx,1))),'--^','color',color_vec(sensor_idx))
                        plot(channel_db_values(scheme_idx,:,agent_db_idx),(squeeze(avg_dep_mse(:,agent_db_idx,:,jj,scheme_idx,sensor_idx,1))),'--x','color',color_vec(sensor_idx))
                        plot(channel_db_values(scheme_idx,:,agent_db_idx),(squeeze(avg_dep_crlb(:,agent_db_idx,:,jj,scheme_idx,sensor_idx,1))),'--o','color',color_vec(sensor_idx))
                    end
                    all_vals = [all_vals; squeeze(avg_dep_var(:,agent_db_idx,:,jj,scheme_idx,sensor_idx,1));...
                    squeeze(avg_dep_mse(:,agent_db_idx,:,jj,scheme_idx,sensor_idx,1));...
                    squeeze(avg_dep_crlb(:,agent_db_idx,:,jj,scheme_idx,sensor_idx,1))];
                end
            end
            
            if experiment == "agent_server_noise"
                xlabel('$$\Gamma$$','Interpreter','latex')
            else
                xlabel('Channel SNR (dB)','Interpreter','latex')
            end
            ylabel('$$\mathrm{E}\{(\hat{'+params(jj)+'}-'+params(jj)+')^2\}$$','Interpreter','latex')
            if experiment == "random_dropout"
                legend(["Drop = " + dropout_vals,"VAR","MSE","CRLB"])
            else
                legend(["S = " + sensor_vals,"VAR","MSE","CRLB"])
            end
            
            % if (jj == 1 && experiment ~= "random_dropout")
            %     ylim([-0.5,1.1*(max(all_vals))])
            % else
            %     ylim([-0.1,1.1*(max(all_vals))])
            % end

            title(params(jj) + " Estimation, " + selected_schemes(scheme_idx))
        end
        % sgtitle(fig1,"$$\mathrm{Deployments} = " + nDeployments + ",\, \mathrm{Trials} = " + nTrials + ",\, S = " + S + ",\, K = " + K + ",\,$$ Sensor SNR $$ = " + agent_db_values(agent_db_idx) + "\ \mathrm{dB},\, \alpha =" + alpha_true + ",\, t_0 = " + t0_true + ",\, m_i \sim U[" + lower_m + "," + upper_m +"],\, \tau_i \sim U[" + 0 + "," + max_tau +"]$$","interpreter","latex")
    end
end
%% Functions


% given the single-sided FFT, get the ifft and corresponding time-shifted
% matrix
function [Qn,Qn_matrix] = get_time_domain(mag_sqr_H,dt,nDeployments)
    mag_sqr_H_2 = [mag_sqr_H(1,1,:) mag_sqr_H(1,2:end,:) fliplr(conj(mag_sqr_H(1,2:end,:)))];
    Qn = ifft(mag_sqr_H_2/dt,[],2);
    L = floor(length(Qn)/2);
    mirrored_Qn = [Qn(1,L+2:end,:), Qn(1,1:L+1,:)];
    Qn_matrix = zeros(L+1,L+1,nDeployments);
    for i = 1:(L+1)
        Qn_matrix(i,:,:) = mirrored_Qn(1,L+1-(i-1):end-(i-1),:);
    end
end

function u = calculate_u(t,xi_tau,m_sensors,tau,dtau)
    S = length(m_sensors);
    nTrials = size(xi_tau,3);
    nDeployments = size(xi_tau,4);

    % y = sum(channel_gains .* sum(xi_tau .* m_sensors .* sensor_signal(tau-t),1)*dtau,2) + n;
    tempC = pagemtimes(pagetranspose(xi_tau),m_sensors .* sensor_signal(tau-t));
    diag_idx = sub2ind(size(tempC),1:S,1:S);
    KTD = size(xi_tau,3)*size(xi_tau,4);
    all_diag_idx = diag_idx + S^2 *reshape(0:(KTD-1),1,1,nTrials,nDeployments);

    aa = tempC(all_diag_idx)*dtau;
    disp('')
end

function y = calculate_y(t,xi_tau,m_sensors,tau,n,dtau,channel_gains)
    S = length(m_sensors);
    nTrials = size(xi_tau,3);
    nDeployments = size(xi_tau,4);

    % y = sum(channel_gains .* sum(xi_tau .* m_sensors .* sensor_signal(tau-t),1)*dtau,2) + n;
    tempC = pagemtimes(pagetranspose(xi_tau),m_sensors .* sensor_signal(tau-t));
    diag_idx = sub2ind(size(tempC),1:S,1:S);
    KTD = size(xi_tau,3)*size(xi_tau,4);
    all_diag_idx = diag_idx + S^2 *reshape(0:(KTD-1),1,1,nTrials,nDeployments);

    aa = tempC(all_diag_idx)*dtau;
    y = pagemtimes(aa,pagetranspose(channel_gains)) + n;
    disp('')
end

function [out] = matched_filter_integral(tempA, tempB, channel_gains, K, S, nTrials, nDeployments, dt)
    % tempA = m_sensors .* reshape(sensor_signal(t-tau_sensors-t0_estimates),K,S,1,nTrials,nDeployments);
    % tempB = m_sensors .* sensor_signal(t-tau_sensors-reshape(t,1,1,length(t)));
    tempC = pagemtimes(pagetranspose(tempA),tempB);

    diag_idx = sub2ind(size(tempC),1:S,1:S);
    KTD = K*nTrials*nDeployments;
    all_diag_idx = diag_idx + S^2 *reshape(0:(KTD-1),1,1,K,nTrials,nDeployments);

    tempD = tempC(all_diag_idx)*dt;
    out = pagemtimes(tempD,reshape(channel_gains,S,1,1,1,nDeployments));
end

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
    