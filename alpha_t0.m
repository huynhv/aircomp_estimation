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
experiment = "disjoint";
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
    
                            pR = scaled_w_psd_constant * sum(m_sensors.^2 .* (a.*real_channel - imag_channel).^2, 2);
                            qR = (scaled_n_psd_constant/2 * (a.^2 + 1)) - gamma*pR;
    
                            pI = scaled_w_psd_constant * sum(m_sensors.^2 .* (real_channel + a.*imag_channel).^2, 2);
                            qI = (scaled_n_psd_constant/2 * (a.^2 + 1)) - gamma*pI;
    
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
                            % aaa represents sum gi int si(tau - t0_estimates) si(tau-t) dtau
                            % each trial will have a different t0 estimate
                            %
                            aaa = reshape(matched_filter_integral(reshape(m_sensors .* sensor_signal(t-tau_sensors-t0_estimates),K,S,1,nTrials,nDeployments), reshape(m_sensors .* sensor_signal(t-tau_sensors-reshape(t,1,1,length(t))), K,S,K,1,nDeployments), channel_gains, K, S, nTrials, nDeployments, dt),1,K,nTrials,nDeployments);
    
                            aaaR = real(aaa);
                            aaaI = imag(aaa);
    
                            % the d is decorrelation, not derivative
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
                            C1 = Beta .* scaled_w_psd_constant;
                            C2 = scaled_n_psd_constant/2;
    
                            V = Beta .* scaled_w_psd_constant .*(1+cos(omega)) ./ (omega.^2 - pi^2).^2 + scaled_n_psd_constant/2;
                                
                            mag_sqr_H = 1./V;
                
                            for idx=indet_idx
                                % fill by L'Hopital's
                                disp('filling indeterminate values')
                                mag_sqr_H(1,idx,:) = 8*pi^2 / (C2*8*pi^2 + C1);
                            end
    
                            [~,Qn_matrix] = get_time_domain(mag_sqr_H,dt,nDeployments);
                            % reshape(m_sensors .* sensor_signal(t-tau_sensors-reshape(t,1,1,length(t))), K,S,K,1,nDeployments)
                            Omega = matched_filter_integral(reshape(m_sensors .* sensor_signal(t-tau_sensors-t0_estimates),K,S,1,nTrials,nDeployments), reshape(m_sensors .* sensor_signal(t-tau_sensors-reshape(t,1,1,length(t))), K,S,K,1,nDeployments), channel_gains, K, S, nTrials, nDeployments, dt);
                        
                            num = dt*dt*pagemtimes(pagemtimes(real(y),reshape(Qn_matrix,length(t),length(t),1,nDeployments)),pagetranspose(reshape(Omega,1,length(t),nTrials,nDeployments)));
                            denom = dt*dt*pagemtimes(pagemtimes(reshape(Omega,1,length(t),nTrials,nDeployments),reshape(Qn_matrix,length(t),length(t),1,nDeployments)),pagetranspose(reshape(Omega,1,length(t),nTrials,nDeployments)));
                        end
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
                    bias(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:) = mean(t0_estimates_for_plot,3);
    
                    %%%%%%%%%%%%%%%%% CRLB %%%%%%%%%%%%%%%%%
                    % Calculate the CRLBs
                    dsalpha = reshape(squeeze(sum(channel_gains .* sum(m_sensors .* sensor_signal(t-tau_sensors-t0_true) .* m_sensors .* sensor_signal(t-tau_sensors-reshape(t,1,1,length(t))) * dt,1),2)),K,1,nDeployments);
                    temp = -w0*cos(w0*(t-tau_sensors-t0_true)).*(heaviside(t-tau_sensors-t0_true) - heaviside(t-tau_sensors-t0_true-T_s));
                    dst0 = reshape(squeeze(sum(channel_gains .* sum(alpha_true .* m_sensors.^2 .* temp .* sensor_signal(t-tau_sensors-reshape(t,1,1,length(t))) * dt,1),2)),K,1,nDeployments);
                    
                    if exp_split(1) == "naive"
                        crlb(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:) = scaled_w_psd_constant / sum(E_i,2);
                        temp = -w0*cos(w0*(t-tau_sensors-t0_true)).*(heaviside(t-tau_sensors-t0_true) - heaviside(t-tau_sensors-t0_true-T_s)) + ...
                        sin(w0*(t-tau_sensors-t0_true)).*-1.*(dirac(t-tau_sensors-t0_true) - dirac(t-tau_sensors-t0_true-T_s));
                        crlb(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:) = scaled_w_psd_constant / sum(sum((alpha_true .* m_sensors .* temp).^2 * dt,1),2);

                        E_i = sum(abs(s_i_true).^2 .* dt,1); 
                        % closed-form variance for alpha
                        my_var(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:) = (sum(real(channel_gains).^2 .* E_i,2) .* scaled_w_psd_constant + n_psd_function_td/2 ) ./ (sum(real(channel_gains).*E_i,2).^2);
                    elseif (scheme == "NPC" || scheme == "PPC")
                        %%% for alpha
                        dsdR_alpha = dt*(pagemtimes(pagetranspose(real(dsalpha)),hdR_matrix) - pagemtimes(pagetranspose(imag(dsalpha)),hdI_matrix));
                        dsdI_alpha = dt*(pagemtimes(pagetranspose(imag(dsalpha)),hdR_matrix) + pagemtimes(pagetranspose(real(dsalpha)),hdI_matrix));

                        %%% for t0
                        dsdR_t0 = dt*(pagemtimes(pagetranspose(real(dst0)),hdR_matrix) - pagemtimes(pagetranspose(imag(dst0)),hdI_matrix));
                        dsdI_t0 = dt*(pagemtimes(pagetranspose(imag(dst0)),hdR_matrix) + pagemtimes(pagetranspose(real(dst0)),hdI_matrix));

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
% elseif experiment == "naive_joint"
%     save("naive_joint.mat")
% elseif experiment == "naive_disjoint"
%     save("naive_disjoint.mat")
% else
%     disp("Unknown experiment specified. Results not saved!")
% end
% 
% %% save bias
% avg_bias = mean(squeeze(bias),5);
% save("new_"+experiment+"_bias"+".mat",'avg_bias');

%% avg linear plots

close all

if nDeployments == 1
    avg_dep_var = empirical_var;
    avg_dep_mse = empirical_mse;
    avg_dep_crlb = crlb;
    avg_dep_my_var = my_var;
else
    avg_dep_var = mean(empirical_var,length(size(empirical_var)));
    avg_dep_mse = mean(empirical_mse,length(size(empirical_mse)));
    avg_dep_crlb = mean(crlb,length(size(crlb)));
    avg_dep_my_var = mean(my_var,length(size(crlb)));
end

params_latex = ["\alpha","t_0"];
params_text = ["alpha","t0"];
color_vec = ["#333333", "#A2142F", "#0072BD"];

sensor_vals_for_plot = sensor_vals;

for agent_db_idx = 1:length(agent_db_values)
    %%% save
    % save_folder = "C:\Users\Vincent Huynh\Desktop\Diff mi ti AirComp Results\" + experiment + "_" + string(agent_db_values(agent_db_idx)) + "_db"; % Define save folder
    % desktop_path = "C:\Users\Vincent Huynh\Desktop";
    % if ~exist(save_folder, 'dir') % Check if the folder exists
    %     mkdir(save_folder); % Create the folder if it doesn't exist
    % end
    %%%

    % fig1 = figure;
    % set(fig1,'WindowState', 'maximized');
    % tiledlayout(2,4, "TileSpacing", "compact","Padding","tight");
    % 
    for param_idx = 1:2
        all_vals = [avg_dep_var(:,agent_db_idx,:,param_idx,:,1:length(sensor_vals));...
                    avg_dep_mse(:,agent_db_idx,:,param_idx,:,1:length(sensor_vals));...
                    avg_dep_crlb(:,agent_db_idx,:,param_idx,:,1:length(sensor_vals))];
        for scheme_idx = 1:length(selected_schemes)
            fig1 = figure;
            set(fig1,'Position',[100,100,450,450])

            nexttile
            hold on

            if experiment == "random_dropout"
                for dropout_idx = 1:length(dropout_vals)
                    plot(nan,nan,'color',color_vec(dropout_idx));
                end
            else
                for sensor_idx = 1:length(sensor_vals_for_plot) 
                    plot(nan,nan,'color',color_vec(sensor_idx));
                end
            end
            plot(nan,nan,'^','color','black');
            plot(nan,nan,'x','color','black');
            plot(nan,nan,'o','color','black');

            % if ndims(channel_db_values) == 2
            %     channel_db_for_plot = channel_db_values;
            % else
            %     channel_db_for_plot = channel_db_values(scheme_idx,:,agent_db_idx);
            % end
            channel_db_for_plot = channel_db_values(scheme_idx,:,agent_db_idx);

            if experiment == "random_dropout"
                for dropout_idx = 1:length(dropout_vals)
                    plot(channel_db_for_plot,(squeeze(avg_dep_var(:,agent_db_idx,:,param_idx,scheme_idx,dropout_idx,1))),'-^','color',color_vec(dropout_idx),'LineWidth', 1)
                    plot(channel_db_for_plot,(squeeze(avg_dep_mse(:,agent_db_idx,:,param_idx,scheme_idx,dropout_idx,1))),'-x','color',color_vec(dropout_idx),'LineWidth', 1)
                    plot(channel_db_for_plot,(squeeze(avg_dep_crlb(:,agent_db_idx,:,param_idx,scheme_idx,dropout_idx,1))),'--o','color',color_vec(dropout_idx),'LineWidth', 1)
                    % all_vals = [all_vals; squeeze(avg_dep_var(:,agent_db_idx,:,param_idx,scheme_idx,dropout_idx,1));...
                    % squeeze(avg_dep_mse(:,agent_db_idx,:,param_idx,scheme_idx,dropout_idx,1));...
                    % squeeze(avg_dep_crlb(:,agent_db_idx,:,param_idx,scheme_idx,dropout_idx,1))];
                end
            else
                for sensor_idx = 1:length(sensor_vals_for_plot) 
                    if experiment == "agent_server_noise"
                        plot(ratio_arr,(squeeze(avg_dep_var(:,agent_db_idx,:,param_idx,scheme_idx,sensor_idx,1))),'-^','color',color_vec(sensor_idx),'LineWidth', 1)
                        plot(ratio_arr,(squeeze(avg_dep_mse(:,agent_db_idx,:,param_idx,scheme_idx,sensor_idx,1))),'-x','color',color_vec(sensor_idx),'LineWidth', 1)
                        plot(ratio_arr,(squeeze(avg_dep_crlb(:,agent_db_idx,:,param_idx,scheme_idx,sensor_idx,1))),'--o','color',color_vec(sensor_idx),'LineWidth', 1)
                    else
                        plot(channel_db_for_plot,(squeeze(avg_dep_var(:,agent_db_idx,:,param_idx,scheme_idx,sensor_idx,1))),'-^','color',color_vec(sensor_idx),'LineWidth', 1)
                        plot(channel_db_for_plot,(squeeze(avg_dep_mse(:,agent_db_idx,:,param_idx,scheme_idx,sensor_idx,1))),'-x','color',color_vec(sensor_idx),'LineWidth', 1)
                        plot(channel_db_for_plot,(squeeze(avg_dep_crlb(:,agent_db_idx,:,param_idx,scheme_idx,sensor_idx,1))),'--o','color',color_vec(sensor_idx),'LineWidth', 1)
                        plot(channel_db_for_plot,(squeeze(avg_dep_my_var(:,agent_db_idx,:,param_idx,scheme_idx,sensor_idx,1))),'--square','color',color_vec(sensor_idx),'LineWidth', 1)
                    end
                    % all_vals = [all_vals; squeeze(avg_dep_var(:,agent_db_idx,:,param_idx,scheme_idx,sensor_idx,1));...
                    % squeeze(avg_dep_mse(:,agent_db_idx,:,param_idx,scheme_idx,sensor_idx,1));...
                    % squeeze(avg_dep_crlb(:,agent_db_idx,:,param_idx,scheme_idx,sensor_idx,1))];
                end
            end

            if experiment == "agent_server_noise"
                xlabel('$$\Gamma$$','Interpreter','latex')
            else
                xlabel('Channel SNR (dB)')
            end
            ylabel(" ")
            if experiment == "random_dropout"
                legend(["Drop = " + dropout_vals,"VAR","MSE","CRLB"], 'Location', 'best')
            else
                legend(["S = " + sensor_vals_for_plot(1:end),"VAR","MSE","CRLB"], 'Location', 'best')
            end

            % title('$$\hat{'+params(jj)+'}$$, ' + selected_schemes(scheme_idx),'Interpreter','latex')
            title('$$\hat{'+params_latex(param_idx)+'}$$','Interpreter','latex')
            set(gca, 'FontSize', 15);
            set(gca, 'YScale', 'log')

            ylim([10^(-0.25)*min(all_vals,[],"all"),1.05*max(all_vals,[],"all")]);

            %%% Save the figure
            % exportgraphics(fig1, fullfile(save_folder, sprintf(experiment + "_" + string(agent_db_values(agent_db_idx)) + "_db_" + params_text(param_idx) + "_" + selected_schemes(scheme_idx) + ".png")), 'Resolution', 300);
        end
    end
end

%% Plot bias
% params_latex = ["\alpha","t_0"];
% params_text = ["alpha","t0"];
% color_vec = ["#006400", "#FF8C00", "#9400D3"];
% 
% sensor_vals_for_plot = sensor_vals;
% 
% close all
% 
% for agent_db_idx = 1:length(agent_db_values)
%     % save_folder = "C:\Users\vhuyn\Desktop\Diff mi ti AirComp Results\" + "naive_bias_" + string(agent_db_values(agent_db_idx)) + "_db"; % Define save folder
%     % if ~exist(save_folder, 'dir') % Check if the folder exists
%     %     mkdir(save_folder); % Create the folder if it doesn't exist
%     % end
% 
%     for param_idx = 1:2
%         if param_idx == 1
%             param = alpha_true;
%         else
%             param = t0_true;
%         end
% 
%          all_vals = [dag(:,param_idx,:,2:length(sensor_vals))-param;...
%                     jag(:,param_idx,:,2:length(sensor_vals))-param];
% 
%         for scheme_idx = 1:length(selected_schemes)
%             fig1 = figure;
%             set(fig1,'Position',[100,100,450,450])
% 
%             nexttile
%             hold on
% 
%             for sensor_idx = 2:length(sensor_vals_for_plot) 
%                 plot(nan,nan,'color',color_vec(sensor_idx));
%             end
% 
%             plot(nan,nan,'^','color','black');
%             plot(nan,nan,'square','color','black');
% 
%             if ndims(channel_db_values) == 2
%                 channel_db_for_plot = channel_db_values;
%             else
%                 channel_db_for_plot = channel_db_values(scheme_idx,:,agent_db_idx);
%             end
% 
%             for sensor_idx = 2:length(sensor_vals_for_plot) 
%                 plot(dag(:,param_idx,scheme_idx,sensor_idx)-param,'-^','color',color_vec(sensor_idx),'LineWidth', 1)
%                 plot(jag(:,param_idx,scheme_idx,sensor_idx)-param,'-square','color',color_vec(sensor_idx),'LineWidth', 1)
%             end
% 
%             if experiment == "agent_server_noise"
%                 xlabel('$$\Gamma$$','Interpreter','latex')
%             else
%                 xlabel('Channel SNR (dB)')
%             end
% 
%             if param_idx == 1
%                 title('$$\hat{\alpha} - \alpha$$','Interpreter','latex')
%             else
%                 title('$$\hat{t}_0 - t_0$$','Interpreter','latex')
%             end
% 
%             legend(["S = " + sensor_vals_for_plot(2:end),"Disjoint","Joint"])
% 
%             % title('$$\hat{'+params(jj)+'}$$, ' + selected_schemes(scheme_idx),'Interpreter','latex')
%             % title('$$\hat{'+params_latex(param_idx)+'}$$','Interpreter','latex')
%             set(gca, 'FontSize', 15);
%             % set(gca, 'YScale', 'log')
% 
%             ylim([min(all_vals,[],"all")-0.05,1.05*max(all_vals,[],"all")]);
% 
% 
%             % Save the figure
%             % exportgraphics(fig1, fullfile(save_folder, sprintf("naive_bias_" + string(agent_db_values(agent_db_idx)) + "_db_" + params_text(param_idx) + "_" + selected_schemes(scheme_idx) + ".png")), 'Resolution', 300);
%         end
%     end
% end

%% Functions
% function [out] = mf_intergral_over_deps(tempA, tempB, channel_gains, K, S, nTrials, nDeployments, dt)

function [out] = matched_filter_integral(tempA, tempB, channel_gains, K, S, nTrials, nDeployments, dt)
    tempC = pagemtimes(pagetranspose(tempA),tempB);

    diag_idx = sub2ind(size(tempC),1:S,1:S);
    KTD = K*nTrials*nDeployments;
    all_diag_idx = diag_idx + S^2 *reshape(0:(KTD-1),1,1,K,nTrials,nDeployments);
    % 
    % tempD = zeros(1, S, K, nTrials, nDeployments);
    tempD = tempC(all_diag_idx)*dt;
    out = pagemtimes(tempD,reshape(channel_gains,S,1,1,1,nDeployments));
end

function [Qn,Qn_matrix] = get_time_domain(mag_sqr_H,dt,nDeployments)
    % 5/17/25: added the 1/2 factor for both sides of mag_sqr_H_2
    mag_sqr_H_2 = [mag_sqr_H(1,1,:) mag_sqr_H(1,2:end,:) fliplr(conj(mag_sqr_H(1,2:end,:)))];
    Qn = ifft(mag_sqr_H_2/dt,[],2);
    L = floor(size(Qn,2)/2);
    mirrored_Qn = [Qn(1,L+2:end,:), Qn(1,1:L+1,:)];
    Qn_matrix = zeros(L+1,L+1,nDeployments);
    for i = 1:(L+1)
        Qn_matrix(i,:,:) = mirrored_Qn(1,L+1-(i-1):end-(i-1),:);
    end
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
    