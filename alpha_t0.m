% Clear all variables and close all exisiting figures.
clear all
close all

%%% 01-18-2026 - NOTE: At the moment, tpi experiments are currently non-functioning due to updates in the matched filtering code. Please refer to repo version from Sept. 7, 2025 submission if you are interested in testing out the framework with imperfect time synchronization.
%%%                    Additionally, the tpi experiment results were only included as ablation study results in the the response letter and not included in the manuscript, hence they have not been updated in this most recent submission.

% Choose which experiment to run.
% "oma_disjoint", "oma_joint", "nae_disjoint", "nae_joint", "cwe_disjoint", "cwe_joint", "cwe_joint_grid", "random_dropout", "nae_joint_imperfect_tpi", "cwe_joint_imperfect_tpi"
experiment_list = ["oma_disjoint", "oma_joint", "nae_disjoint", "nae_joint", "cwe_disjoint", "cwe_joint", "cwe_joint_grid", "random_dropout", ...
                    "comp_cwe_disjoint", "comp_cwe_joint"];
experiment = "oma_disjoint";

msg = ['Running ' char(experiment)];
border = repmat('─', 1, length(msg)+4);

fprintf('\n╭%s╮\n', border);
fprintf('│  %s  │\n', msg);
fprintf('╰%s╯\n\n', border);

% Set a seed for reproducibility.
seed = 264;
rng(seed);

% Set number of deployments.
nDeployments = 100;
% Set number of trials.
nTrials = 500;
% Set the number of measurements per sensor. The sample index starts at 0 so
% sampling from index 0 to 100 requires K = 101 measurements.
K = 301;

% Set some bookkeeping variables depending on which experiment we are running.
if experiment == "random_dropout"
    dropout_vals = [0,2,4];
    agent_db_values = 0;
    sensor_dimension = length(dropout_vals);
    % Here we only consider maximum number of sensors and drop from there.
    sensor_vals = 10;
else
    if experiment == "cwe_joint_grid"
        sensor_vals = 5;
        agent_db_values = [0,5,10,15,20];
    elseif contains(experiment, "oma")
        sensor_vals = [2,4,6];
        agent_db_values = 0;
    elseif contains(experiment, "comp")
        sensor_vals = [2,4,6];
        agent_db_values = 0;
    else
        sensor_vals = [5,10];
        agent_db_values = 0;
    end
    dropout_vals = 0;
    sensor_dimension = length(sensor_vals);
end

% Parse experiment name for organization.
% exp_split = strsplit(experiment,'_');

% Set observation window length in seconds.
T0 = 5;

% Define event signal parameters.
Tp = 1;
w0 = pi;

% Define time domain parameters
dt = T0/(K-1);
t = dt*(0:K-1)';

% Define frequency domain parameters
fs = 1/dt;
L = K-1;
f = 0:fs/(2*L):fs/2;
omega = 2*pi*f;

% Calculate the signal energy (such that the signal is assumed to be within the
% interioir fo the observation period).
func = @(t) abs(sensor_signal(t)).^2;
P_s = integral(func,0,T0)/T0;

% Define true parameter values.
alpha_true = 2;
t0_true = 1;

% Define range for distribution of mi and ti (tau_i)
max_mi = 1;
min_mi = 0.5;

max_ti = 1;
min_ti = 0;

% Set max tpi value based on sampling rate to give a frame of reference as to
% how many sample the error may shift the peak
max_tpi = 0.02;
min_tpi = 0;

% Set maximum value of t0 such that the signal from each sensor is guaranteed to
% be in the interioir of the observation period.
t0_max = T0-Tp-max_ti-max_tpi;

%%% Always generate w, n, and channel gain for S = 10, then use subsets for different S trials
max_sensor_limit = 10;
% Generate sensor noise with dimensions: K x S x trials x deployments.
all_w = randn(K,max_sensor_limit,nTrials,nDeployments);
% Generate server noise.
n = (randn(K,max_sensor_limit,nTrials,nDeployments) + 1i*randn(K,max_sensor_limit,nTrials,nDeployments));
% Generate channel gains.
all_gi = (randn(1,max_sensor_limit,1,nDeployments) + 1i*randn(1,max_sensor_limit,1,nDeployments))/sqrt(2);

% Generate mi and ti.
all_mi = unifrnd(min_mi,max_mi,1,max_sensor_limit,1,nDeployments);
all_ti = unifrnd(min_ti,max_ti,1,max_sensor_limit,1,nDeployments);

% Choose which schemes we want to plot. Note that although we still generate the
% results for NPC, we do not necessarily plot them. there is no gurantee that
% more sensors will perform better for NPC due to the uncompensated channel
% phase.
selected_schemes = ["MPC","EPC","PPC","NPC"];

% Generate clock offset vector if needed
if experiment == "nae_joint_imperfect_tpi" || experiment == "cwe_joint_imperfect_tpi"
    all_tpi = unifrnd(min_tpi,max_tpi,1,max(sensor_vals),1,nDeployments);
else
    all_tpi = zeros(1,max(sensor_vals),1,nDeployments);
end

%%% 10-5-2025
%%% Need to do fresh push that states we addressed the eratta
if experiment == "random_dropout"
    which_dropout = zeros(nDeployments,max(dropout_vals));
    for idx = 1:nDeployments
        which_dropout(idx,:) = randperm(max(sensor_vals),max(dropout_vals));
    end
end

% Define Rayleigh distribution parameters.
rayleigh_factor = 1/sqrt(2);
E_mag_g_sqr = 2*rayleigh_factor^2;

% Define expected value of mi^2.
E_mi_sqr = (1/12) * (max_mi-min_mi)^2 + 0.5*(min_mi+max_mi);

% Define channel snr values.
channel_snr = [0,5,10,15,20];
% Duplicate channel snr matrices for experiments.
channel_db_values = repmat(channel_snr, length(selected_schemes), 1, length(agent_db_values));

% Initialize empty matrices for crlb, variance, mse, and bias. The parameters
% are organized such that alpha is index 1, t0 is index 2. The dimensions of
% crlb are: 1 x agent snr x channel snr x parameter x scheme x num sensors x
% deployments.
crlb = zeros(1,length(agent_db_values),size(channel_db_values,2),2,length(selected_schemes),sensor_dimension,nDeployments);
empirical_var = crlb;
empirical_mse = crlb;
empirical_bias = crlb;
my_var = crlb;

% Start run timer.
loopTic = tic;

%%% debugging
% save_xi_check = true;
%%%

% Iterate through agent snr values.
for agent_db_idx = 1:length(agent_db_values)
    agent_db = agent_db_values(agent_db_idx);
    
    % Set sensor noise power. Note the factor of 1/dt, which is equivalent to
    % applying an anti-aliasing filter.
    w_psd_constant = 1;
    gamma_w = (dt/w_psd_constant) * (P_s / db2magTen(agent_db));
    scaled_w = sqrt(gamma_w * w_psd_constant /dt) * all_w; % --> variance of this should be gamma_w/dt
    scaled_w_psd_constant = gamma_w * w_psd_constant; %% = Nw/2

    % Iterate through sensor values.
    for sensor_idx = 1:length(sensor_vals)
        S = sensor_vals(sensor_idx);

        % Iterate through dropout values.
        for dropout_idx = 1:length(dropout_vals)
            num_dropout = dropout_vals(dropout_idx);
            
            % Select subset of mi, ti, and gi values.
            mi = all_mi(:,1:S,:,:);
            ti = all_ti(:,1:S,:,:);
            tpi = all_tpi(:,1:S,:,:);
            gi = all_gi(:,1:S,:,:);

            % Calculate quantized phase.
            M = 4;
            [qgi,~] = quantize_comp(gi,M);
           
            % Generate si(t-t0).
            si_true = mi .* sensor_signal(t - ti - t0_true); % K x S
            % Generate xi(t).
            xi = alpha_true * si_true + scaled_w(:,1:S,:,:); % K x S x nTrials x nDeployments
            % Compute Ei.
            Ei = sum(abs(si_true).^2 .* dt,1); % 1 x S

            noisy_unified_xi = alpha_true .* mi .* sensor_signal(t-t0_true) + scaled_w(:,1:S,:,:);
            
            %%% debugging
            % if save_xi_check
            %     save(experiment + "_xi.mat", "xi", "si_true")
            %     save_xi_check = false;
            % end
            %%%

            % Iterate through channel snr values.
            for channel_db_idx = 1:size(channel_db_values,2)
                % Iterate through schemes.
                for scheme_idx = 1:length(selected_schemes)
                    scheme = selected_schemes(scheme_idx);

                    % Print statement for at-a-glance performance.
                    disp('')
                    disp("=== " + scheme + " ===")
                    disp("** Agent SNR = " + (agent_db_values(agent_db_idx)) + " dB **")
                    disp("** Channel SNR = " + (channel_db_values(scheme_idx,channel_db_idx,agent_db_idx)) + " dB **")
                    disp("** S = " + S + ", Dropout = " + dropout_vals(dropout_idx) + " **")

                    % Define channel gains and server noise power.
                    if scheme == "MPC"
                        ci = ones(size(gi));
                        gamma_n = dt * P_s * E_mi_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                    elseif scheme == "EPC"
                        ci = abs(gi);
                        gamma_n = dt * P_s * E_mi_sqr * E_mag_g_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                    elseif scheme == "PPC"
                        ci = qgi;
                        gamma_n = dt * P_s * E_mi_sqr * E_mag_g_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                    elseif scheme == "NPC"
                        ci = gi;
                        gamma_n = dt * P_s * E_mi_sqr * E_mag_g_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                    end

                    % Apply dropout (by setting dropout sensor channels to zero).
                    if num_dropout > 0
                        dropout_elements = which_dropout(:,1:num_dropout);
                        for item = 1:nDeployments
                            ci(1,dropout_elements(item,:),1,item) = 0;
                        end
                    end

                    n_psd_constant = 1;
                    % Set server noise power. Note the factor of 1/dt which is
                    % equivalent to applying an anti-aliasing filter.
                    scaled_n = sqrt( (gamma_n*n_psd_constant/dt) / 2 ) * n;
                    % Define psd function of n in time domain (td).
                    % n_psd_function_td = gamma_n/dt;
                    scaled_n_psd_constant = gamma_n * n_psd_constant; % == N0/2

                    scaled_vi_psd_constant = (abs(ci).^2 * scaled_w_psd_constant + scaled_n_psd_constant/2);

                    %%%%%%%%%%%%%%%%% ESTIMATION %%%%%%%%%%%%%%%%%
                    % Search for t0 from index 1 to index of t0_max/dt + 1.
                    % Since we know a priori that the estimate of t0 cannot
                    % exceed t0_max. If we don't impose this restriction then we
                    % could select a t0 value such that the joint estimate for
                    % alpha = 0/0, which will result in NaN.

                    %% Estimate t0
                    if contains(experiment, "oma")
                        yi_oma = ci .* xi + scaled_n(:,1:S,:,:);
                        yi_oma_tr = pagetranspose(yi_oma);
                        unified_yi = ci .* noisy_unified_xi + scaled_n(:,1:S,:,:);

                        llf_search_space = zeros(1,1,K,nTrials,nDeployments);
                        denom_space = zeros(1,1,K,1,nDeployments);
                        
                        if scheme == "MPC" || scheme == "EPC"
                            % cws_oma = reshape(matched_filter_integral(reshape(real(yi_oma),K,S,1,nTrials,nDeployments), reshape(ci .* mi .* sensor_signal(t-ti-tpi-reshape(t,1,1,K)), K,S,K,1,nDeployments), 1 ./ scaled_vi_psd_constant, K, S, nTrials, nDeployments, dt),1,K,nTrials,nDeployments);
                            % cws_searchable = cws_oma(:,1:(floor(t0_max/dt) + 1),:,:);

                            %%% convolution using FFT checks different offsets for me so I don't have to do it manually
                            [cws_oma, ~] = mf_integral_fft(real(unified_yi), ci .* mi .* sensor_signal(t-tpi), 1 ./ scaled_vi_psd_constant, 1, K, dt, Tp);
                            cws_searchable = cws_oma(1:(floor(t0_max/dt) + 1),:,:,:);
                            
                            [~,I] = max(cws_searchable);
                        
                        elseif scheme == "PPC" || scheme == "NPC"

                            dsalpha = ci .* mi .* sensor_signal(t-ti-t0_true);
                            temp = -w0*cos(w0*(t-ti-t0_true)).*(heaviside(t-ti-t0_true) - heaviside(t-ti-t0_true-Tp)) + sin(w0*(t-ti-t0_true)).*-1.*(dirac(t-ti-t0_true) - dirac(t-ti-t0_true-Tp));
                            dst0 = ci .* alpha_true .* mi .* temp;

                            fisher_matrix_accum = zeros(2,2,nDeployments);

                            real_channel = real(ci);
                            imag_channel = imag(ci);
        
                            % Compute polynomial to find coefficients for
                            % decorrelation filter. 
                            b = (real_channel.*imag_channel) ./ (real_channel.^2 - imag_channel.^2);
                            discr = 1 + 4.*b.^2;
                            a1 = (-1 + sqrt(discr)) ./ (2*b);
                            a2 = (-1 - sqrt(discr)) ./ (2*b);
                            a_select = a1;

                            for i = 1:S
                                cR = real_channel(:,i,:,:);
                                cI = imag_channel(:,i,:,:);
                                a = a_select(:,i,:,:);
                                yi = yi_oma_tr(i,:,:,:); % pagetranspose(yi_oma(:,i,:,:));
                                
                                % Define magnitude squared of SHdI.
                                mag_sqr_SHdI = ones(1,K,1,nDeployments);
    
                                % Define magnitude of SHdI.
                                mag_SHdI = sqrt(mag_sqr_SHdI);
        
                                % Get time domain matrix for hdI.
                                [hdI, hdI_matrix] = get_time_domain(mag_SHdI,dt,nDeployments);
    
                                % Get time domain matrix for hdR.
                                hdR_matrix = reshape(a,1,1,nDeployments) .* hdI_matrix;
    
                                % Create time domain matrix for hd.
                                % hd_matrix = hdR_matrix + 1i*hdI_matrix;
        
                                % Compute pR and qR.
                                pR = scaled_w_psd_constant * (a.*cR - cI).^2;
                                qR = (scaled_n_psd_constant/2 * (a.^2 + 1));
        
                                % Compute pI and qI.
                                pI = scaled_w_psd_constant * (cR + a.*cI).^2;
                                qI = (scaled_n_psd_constant/2 * (a.^2 + 1));
        
                                % Compute SKndR and SKndI.
                                SKndR = pR + mag_sqr_SHdI .* qR;
                                SKndI = pI + mag_sqr_SHdI .* qI;
        
                                % Check to make sure SKndR and SKndI are positive.
                                if sum(SKndR <= 0, "all")
                                    error("SKndR less than or equal to 0!")
                                elseif sum(SKndI <= 0, "all")
                                    error("SKndI less than or equal to 0!")
                                end
        
                                % Generate whitening filters based on inverse
                                % spectrum.
                                SQnR = 1 ./ SKndR;
                                SQnI = 1 ./ SKndI;
            
                                % Get time domain matrix  for QnR and QnI.
                                [~,QnR_matrix] = get_time_domain(SQnR,dt,nDeployments);
                                [~,QnI_matrix] = get_time_domain(SQnI,dt,nDeployments);
        
                                aaa = pagetranspose((cR + 1i*cI) .* mi(:,i,:,:) .* sensor_signal(t-ti(:,i,:,:)-reshape(t,1,1,K)));
                                aaa = reshape(aaa,1,K,K,1,nDeployments);
                                
                                aaaR = real(aaa);
                                aaaI = imag(aaa);

                                hdR_matrix_resh = reshape(hdR_matrix,K,K,1,1,nDeployments);
                                hdI_matrix_resh = reshape(hdI_matrix,K,K,1,1,nDeployments);
        
                                % The d here is for decorrelation, not derivative.
                                % aaadR = dt*(pagemtimes(aaaR,reshape(hdR_matrix,K,K,1,1,nDeployments)) - pagemtimes(aaaI,reshape(hdI_matrix,K,K,1,1,nDeployments)));
                                % aaadI = dt*(pagemtimes(aaaI,reshape(hdR_matrix,K,K,1,1,nDeployments)) + pagemtimes(aaaR,reshape(hdI_matrix,K,K,1,1,nDeployments)));

                                aaadR = dt*(pagemtimes(aaaR,hdR_matrix_resh) - pagemtimes(aaaI,hdI_matrix_resh));
                                aaadI = dt*(pagemtimes(aaaI,hdR_matrix_resh) + pagemtimes(aaaR,hdI_matrix_resh));
        
                                Hc = reshape(hdR_matrix + 1j*hdI_matrix, K, K, 1, nDeployments);
                                yid = dt * pagemtimes(yi, Hc);
                                yidR = real(yid);
                                yidI = imag(yid);

                                resh_QnR = reshape(QnR_matrix,K,K,1,nDeployments);
                                resh_QnI = reshape(QnI_matrix,K,K,1,nDeployments);
        
                                num1 = dt*dt*pagemtimes(reshape(pagemtimes(yidR,resh_QnR), 1,K,1,nTrials,nDeployments),pagetranspose(aaadR));
                                num2 = dt*dt*pagemtimes(reshape(pagemtimes(yidI,resh_QnI), 1,K,1,nTrials,nDeployments),pagetranspose(aaadI));

                                resh_QnR_matrix2 = reshape(QnR_matrix,K,K,1,1,nDeployments);
                                resh_QnI_matrix2 = reshape(QnI_matrix,K,K,1,1,nDeployments);

                                denom1 = dt*dt*pagemtimes(pagemtimes(aaadR,resh_QnR_matrix2),pagetranspose(aaadR));
                                denom2 = dt*dt*pagemtimes(pagemtimes(aaadI,resh_QnI_matrix2),pagetranspose(aaadI));

                                llf_search_space = llf_search_space + num1 + num2;
                                denom_space = denom_space + denom1 + denom2;

                                %% also compute CRLB quantities while we are here
                                dsalpha_i = reshape(dsalpha(:,i,:,:),K,1,nDeployments);
                                dst0_i = reshape(dst0(:,i,:,:),K,1,nDeployments);

                                % Compute decorrelated derivative expressions for alpha.
                                dsdR_alpha = dt*(pagemtimes(pagetranspose(real(dsalpha_i)),hdR_matrix) - pagemtimes(pagetranspose(imag(dsalpha_i)),hdI_matrix));
                                dsdI_alpha = dt*(pagemtimes(pagetranspose(imag(dsalpha_i)),hdR_matrix) + pagemtimes(pagetranspose(real(dsalpha_i)),hdI_matrix));
        
                                % Compute decorrelated derivative expressions for t0.
                                dsdR_t0 = dt*(pagemtimes(pagetranspose(real(dst0_i)),hdR_matrix) - pagemtimes(pagetranspose(imag(dst0_i)),hdI_matrix));
                                dsdI_t0 = dt*(pagemtimes(pagetranspose(imag(dst0_i)),hdR_matrix) + pagemtimes(pagetranspose(real(dst0_i)),hdI_matrix));
        
                                % % Compute fisher information matrix (FIM).
                                fisher_matrix_accum = fisher_matrix_accum + dt*dt*[(pagemtimes(pagemtimes(dsdR_alpha,QnR_matrix),pagetranspose(dsdR_alpha)) + pagemtimes(pagemtimes(dsdI_alpha,QnI_matrix),pagetranspose(dsdI_alpha)))...
                                            (pagemtimes(pagemtimes(dsdR_alpha,QnR_matrix),pagetranspose(dsdR_t0)) + pagemtimes(pagemtimes(dsdI_alpha,QnI_matrix),pagetranspose(dsdI_t0)));...
                                            (pagemtimes(pagemtimes(dsdR_alpha,QnR_matrix),pagetranspose(dsdR_t0)) + pagemtimes(pagemtimes(dsdI_alpha,QnI_matrix),pagetranspose(dsdI_t0)))...
                                            (pagemtimes(pagemtimes(dsdR_t0,QnR_matrix),pagetranspose(dsdR_t0)) + pagemtimes(pagemtimes(dsdI_t0,QnI_matrix),pagetranspose(dsdI_t0)))];
                            end

                            [~,I] = max(llf_search_space, [], 3);
                            I = reshape(I, 1, 1, nTrials, nDeployments);
                        end

                        t0_estimates_for_plot = (I-1)*dt;

                        if contains(experiment, "_disjoint")
                            t0_estimates_for_alpha = t0_true*ones(1,1,nTrials,nDeployments);
                            I_for_alpha = ((t0_true/dt)+1)*ones(1,1,nTrials,nDeployments);
                        else
                            t0_estimates_for_alpha = t0_estimates_for_plot;
                            I_for_alpha = I;
                        end

                        %% alpha
                        if scheme == "MPC" || scheme == "EPC"
                            num_accum = sum(sum(real(yi_oma) .* mi .* sensor_signal(t-ti-tpi-t0_estimates_for_alpha), 1)*dt .* (ci ./ scaled_vi_psd_constant), 2);
                            denom_accum = sum(abs(ci).^2 ./ scaled_vi_psd_constant .* Ei,2);
                        
                        elseif scheme == "PPC" || scheme == "NPC"
                            num_accum_compare = zeros(1,1,nTrials,nDeployments);
                            denom_accum_compare = zeros(1,1,nTrials,nDeployments);
                        
                            for dep_idx = 1:size(llf_search_space,5)
                                for trial_idx = 1:size(llf_search_space,4)
                                    num_accum_compare(1,1,trial_idx,dep_idx) = llf_search_space(1,1, I_for_alpha(1,1,trial_idx,dep_idx), trial_idx, dep_idx);
                                    denom_accum_compare(1,1,trial_idx,dep_idx) = denom_space(1,1, I_for_alpha(1,1,trial_idx,dep_idx), 1, dep_idx);
                                end
                            end
                        end
                        alpha_estimates = num_accum ./ denom_accum;
                    elseif contains(experiment, "nae") || contains(experiment, "cwe") || experiment == "random_dropout"
                        % Find peak index from noisy channel-weighted sum (cws).
                        % cws = reshape(matched_filter_integral(reshape(xi,K,S,1,nTrials,nDeployments), reshape(mi .* sensor_signal(t-ti-tpi-reshape(t,1,1,K)), K,S,K,1,nDeployments), channel_gains, K, S, nTrials, nDeployments, dt),1,K,nTrials,nDeployments);
                        % y = cws + pagetranspose(scaled_n(:,1,:,:));
                        % y_searchable = y(:,1:(floor(t0_max/dt) + 1),:,:);

                        [cws, ~] = mf_integral_fft(noisy_unified_xi, mi .* sensor_signal(t-tpi), ci, 1, K, dt, Tp);
                        y = pagetranspose(cws + scaled_n(:,1,:,:));
                        y_searchable = y(:,1:(floor(t0_max/dt) + 1),:,:);

                        if contains(experiment, "nae") || scheme == "MPC" || scheme == "EPC"
                            [~,I] = max(abs(real(y_searchable)));
                        else
                            [~,I] = max(abs(y_searchable));
                        end

                        t0_estimates_for_plot = (I-1)*dt;
                        if contains(experiment, "_disjoint") % experiment == "cwe_disjoint" || experiment == "nae_disjoint" || experiment == "oma_disjoint"
                            t0_estimates_for_alpha = t0_true*ones(1,1,nTrials,nDeployments);
                        else
                            t0_estimates_for_alpha = t0_estimates_for_plot;
                        end
                    end
                    
                    
                    %% Estimate alpha
                    if contains(experiment, "oma")
                        % placeholder for oma
                        disp("")

                    elseif contains(experiment, "nae")
                        if contains(experiment, "_joint")
                            peak_y_vals = zeros(1,1,nTrials,nDeployments);

                            for dep_idx = 1:nDeployments
                                for trial_idx = 1:nTrials
                                    peak_y_vals(1,1,trial_idx,dep_idx) = y(1,I(1,1,trial_idx,dep_idx), trial_idx, dep_idx);
                                end
                            end

                        elseif contains(experiment, "_disjoint")
                            % Compute y(t0). We have the index of what we think is the peak value of y (where all the matched filter peaks should line up).
                            peak_y_vals = y(1,(t0_true/dt)+1,:,:,:);
                            
                        end
                        % compute alpha estimates
                        alpha_estimates =  real(peak_y_vals) ./ sum(real(ci) .* Ei,2);

                    % Compute alpha for AC-CWE.
                    elseif contains(experiment, "cwe") || experiment == "random_dropout"
                        % Check if there are any indeterminate values (will
                        % depend on the form of si(t) but in this case we
                        % calculated this ahead of time).
                        indet_idx = find(abs(round(f,4)) == 0.5);

                        % Compute AC-CWE estimates for full phase compensation.
                        if scheme == "MPC" || scheme == "EPC"
                            % Define coefficients in V(w).
                            Beta = 2 * pi^2 * sum(ci.^2 .* mi.^2,2);
                            C1 = Beta .* scaled_w_psd_constant;
                            C2 = scaled_n_psd_constant/2;
    
                            % Compute V(w).
                            V = Beta .* scaled_w_psd_constant .*(1+cos(omega)) ./ (omega.^2 - pi^2).^2 + scaled_n_psd_constant/2;
                                
                            % Compute |H(w)|^2.
                            mag_sqr_H = 1./V;
                
                            % Replace indeterminate values in |H(w)|^2.
                            for idx=indet_idx
                                % Fill indeterminate values using L'Hopital's rule.
                                disp('replacing indeterminate values')
                                mag_sqr_H(1,idx,:) = 8*pi^2 / (C2*8*pi^2 + C1);
                            end
    
                            % Get time domain matrix for Qn.
                            [~, Qn_matrix] = get_time_domain(mag_sqr_H,dt,nDeployments);
                            % Omega = matched_filter_integral(reshape(mi .* sensor_signal(t-ti-t0_estimates),K,S,1,nTrials,nDeployments), reshape(mi .* sensor_signal(t-ti-reshape(t,1,1,K)), K,S,K,1,nDeployments), ci, K, S, nTrials, nDeployments, dt);
                            [Omega, ~] = mf_integral_fft(mi .* sensor_signal(t-t0_estimates_for_alpha), mi .* sensor_signal(t), ci, 1, K, dt, Tp);
                        
                            resh_Qn = reshape(Qn_matrix,K,K,1,nDeployments);
                            resh_Omega = reshape(Omega,1,K,nTrials,nDeployments);

                            num = dt*dt*pagemtimes(pagemtimes(real(y),resh_Qn),pagetranspose(resh_Omega));
                            denom = dt*dt*pagemtimes(pagemtimes(resh_Omega,resh_Qn),pagetranspose(resh_Omega));
                        
                        elseif (scheme == "PPC" || scheme == "NPC")
                            real_channel = real(ci);
                            imag_channel = imag(ci);
        
                            % Compute polynomial to find coefficients for
                            % decorrelation filter. 
                            b = sum(real_channel.*imag_channel.*mi.^2,2) ./ (sum(mi.^2 .* (real_channel.^2 - imag_channel.^2),2));
                            discr = 1 + 4.*b.^2;
                            a1 = (-1 + sqrt(discr)) ./ (2*b);
                            a2 = (-1 - sqrt(discr)) ./ (2*b);
                            a_select = a1;

                            % Define frequency domain expression for s0(t).
                            SEs = ones(1,K,1,nDeployments) .* (2 * pi^2 * (1+cos(omega)) ./ (omega.^2 - pi^2).^2);
                            gamma = 1;
    
                            % Define magnitude squared of SHdI.
                            mag_sqr_SHdI = 1 ./ (gamma + SEs);
    
                            % Replace indeterminate values using L'Hopital's
                            % rule, also will depend on the form of si(t).
                            for idx=indet_idx
                                disp('replacing indeterminate values')
                                % Fill indeterminate values using L'Hopital's
                                % rule.
                                mag_sqr_SHdI(1,idx,:) = 8*w0^2 / (gamma*8*w0^2 - 2*pi^2*cos(w0));
                            end
                            
                            % Define magnitude of SHdI.
                            mag_SHdI = sqrt(mag_sqr_SHdI);
    
                            % Get time domain matrix for hdI.
                            [hdI, hdI_matrix] = get_time_domain(mag_SHdI,dt,nDeployments);

                            % Get time domain matrix for hdR.
                            hdR_matrix = reshape(a_select,1,1,nDeployments) .* hdI_matrix;

                            % Create time domain matrix for hd.
                            % hd_matrix = hdR_matrix + 1i*hdI_matrix;

                            % Compute pR and qR.
                            pR = scaled_w_psd_constant * sum(mi.^2 .* (a_select.*real_channel - imag_channel).^2, 2);
                            qR = (scaled_n_psd_constant/2 * (a_select.^2 + 1)) - gamma*pR;
    
                            % Compute pI and qI.
                            pI = scaled_w_psd_constant * sum(mi.^2 .* (real_channel + a_select.*imag_channel).^2, 2);
                            qI = (scaled_n_psd_constant/2 * (a_select.^2 + 1)) - gamma*pI;
    
                            % Compute SKndR and SKndI.
                            SKndR = pR + mag_sqr_SHdI .* qR;
                            SKndI = pI + mag_sqr_SHdI .* qI;
    
                            % Check to make sure SKndR and SKndI are positive.
                            if sum(SKndR <= 0, "all")
                                error("SKndR less than or equal to 0!")
                            elseif sum(SKndI <= 0, "all")
                                error("SKndI less than or equal to 0!")
                            end
    
                            % Generate whitening filters based on inverse
                            % spectrum.
                            SQnR = 1 ./ SKndR;
                            SQnI = 1 ./ SKndI;
        
                            % Get time domain matrix  for QnR and QnI.
                            [~,QnR_matrix] = get_time_domain(SQnR,dt,nDeployments);
                            [~,QnI_matrix] = get_time_domain(SQnI,dt,nDeployments);
    
                            % Compute closed-form expressions for alpha

                            % aaa represents sum gi int si(tau - t0_estimates) -- si(tau-t) dtau, where each trial will have a different t0 estimate.
                            % aaa = reshape(matched_filter_integral(reshape(mi .* sensor_signal(t-ti-t0_estimates),K,S,1,nTrials,nDeployments), reshape(mi .* sensor_signal(t-ti-reshape(t,1,1,K)), K,S,K,1,nDeployments), channel_gains, K, S, nTrials, nDeployments, dt),1,K,nTrials,nDeployments);
    
                            [aaa, ~] = mf_integral_fft(mi .* sensor_signal(t-t0_estimates_for_alpha), mi .* sensor_signal(t), ci, 1, K, dt, Tp);
                            
                            aaa = pagetranspose(aaa);
                            aaaR = real(aaa);
                            aaaI = imag(aaa);

                            yR = real(y);
                            yI = imag(y);

                            resh_hdR = reshape(hdR_matrix,K,K,1,nDeployments);
                            resh_hdI = reshape(hdI_matrix,K,K,1,nDeployments);

                            resh_QnR = reshape(QnR_matrix,K,K,1,nDeployments);
                            resh_QnI = reshape(QnI_matrix,K,K,1,nDeployments);
    
                            % The d here is for decorrelation, not derivative.
                            aaadR = dt*(pagemtimes(aaaR,resh_hdR) - pagemtimes(aaaI,resh_hdI));
                            aaadI = dt*(pagemtimes(aaaI,resh_hdR) + pagemtimes(aaaR,resh_hdI));
    
                            ydR = dt*(pagemtimes(yR,resh_hdR) - pagemtimes(yI,resh_hdI));
                            ydI = dt*(pagemtimes(yI,resh_hdR) + pagemtimes(yR,resh_hdI));
    
                            num1 = pagemtimes(pagemtimes(ydR,resh_QnR),pagetranspose(aaadR));
                            num2 = pagemtimes(pagemtimes(ydI,resh_QnI),pagetranspose(aaadI));
                            num = dt*dt*(num1+num2);
    
                            denom1 = pagemtimes(pagemtimes(aaadR,resh_QnR),pagetranspose(aaadR));
                            denom2 = pagemtimes(pagemtimes(aaadI,resh_QnI),pagetranspose(aaadI));
                            denom = dt*dt*(denom1+denom2);

                        end
                        alpha_estimates = num ./ denom;
                    end

                    if experiment == "random_dropout"
                        save_dim = dropout_idx;
                    else
                        save_dim = sensor_idx;
                    end

                    % Compute empirical variance.
                    empirical_var(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:) = var(alpha_estimates,0,3);
                    empirical_var(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:) = var(t0_estimates_for_plot,0,3);
    
                    % Compute empirical mse.
                    empirical_mse(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:) = mean((alpha_estimates - alpha_true).^2,3);
                    empirical_mse(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:) = mean((t0_estimates_for_plot - t0_true).^2,3);
    
                    % Compute empirical bias.
                    empirical_bias(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:) = mean(alpha_estimates,3);
                    empirical_bias(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:) = mean(t0_estimates_for_plot,3);
    
                    %%%%%%%%%%%%%%%%% CRLB %%%%%%%%%%%%%%%%%
                    % Compute CRLBs for AC-NAE
                    if contains(experiment, "oma")
                        if scheme == "MPC" || scheme == "EPC"
                            %%% ORTHOG MPC AND EPC %%%
                            % CRLB for alpha
                            crlb(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:) = 1./sum(abs(ci).^2 ./ scaled_vi_psd_constant .* Ei,2);
                            % CRLB for t0
                            temp = -w0*cos(w0*(t-ti-t0_true)).*(heaviside(t-ti-t0_true) - heaviside(t-ti-t0_true-Tp)) + sin(w0*(t-ti-t0_true)).*-1.*(dirac(t-ti-t0_true) - dirac(t-ti-t0_true-Tp));
                            crlb(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:) = 1 ./ sum(sum((alpha_true .* ci .* mi .* temp).^2 * dt,1)./ scaled_vi_psd_constant , 2);
                        elseif scheme == "PPC" || scheme == "NPC"
                            %%% ORTHOG PPC AND NPC %%%
                            inv_zz_accum = pageinv(fisher_matrix_accum);
                            % Compute CRLB for alpha.
                            crlb(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:) = inv_zz_accum(1,1,:);
                            % Compute CRLB for t0.
                            crlb(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:) = inv_zz_accum(2,2,:);
                        end
                            
                    elseif contains(experiment, "nae")
                        % CRLB for alpha
                        crlb(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:) = scaled_w_psd_constant / sum(Ei,2);
                        % CRLB for t0
                        temp = -w0*cos(w0*(t-ti-t0_true)).*(heaviside(t-ti-t0_true) - heaviside(t-ti-t0_true-Tp)) + sin(w0*(t-ti-t0_true)).*-1.*(dirac(t-ti-t0_true) - dirac(t-ti-t0_true-Tp));
                        crlb(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:) = scaled_w_psd_constant / sum(sum((alpha_true .* mi .* temp).^2 * dt,1),2);
                        
                        % compute closed-form variance for alpha (mainly an easy
                        % check to see if theoretical variance matches empirical
                        % variance)
                        % my_var(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:) = (sum(real(channel_gains).^2 .* Ei,2) .* scaled_w_psd_constant + n_psd_function_td/2 ) ./ (sum(real(channel_gains).*Ei,2).^2);
                    
                    % Compute AC-CWE CRLBs for partial or no phase compensation
                    % 9/30/2025 - Add random_dropout logic
                    elseif contains(experiment, "cwe") || experiment == "random_dropout"
                        dsalpha = reshape(squeeze(sum(ci .* sum(mi .* sensor_signal(t-ti-t0_true) .* mi .* sensor_signal(t-ti-reshape(t,1,1,K)) * dt,1),2)),K,1,nDeployments);
                        temp = -w0*cos(w0*(t-ti-t0_true)).*(heaviside(t-ti-t0_true) - heaviside(t-ti-t0_true-Tp)) + sin(w0*(t-ti-t0_true)).*-1.*(dirac(t-ti-t0_true) - dirac(t-ti-t0_true-Tp));
                        dst0 = reshape(squeeze(sum(ci .* sum(alpha_true .* mi.^2 .* temp .* sensor_signal(t-ti-reshape(t,1,1,K)) * dt,1),2)),K,1,nDeployments);
                        
                        alpha_objective_array = @(ai) 1./( ((sum(ci .* ai .* mi.^2)).^2 / (2*pi)) .* integral(@(w) (mag_sqr_S0(w)).^2 / V_arr(w,ai), -Inf, Inf, 'ArrayValued', true));
                        t0_objective_array = @(ai) 1./( (alpha_true^2 .* (sum(ci .* ai .* mi.^2)).^2 / (2*pi)) .* integral(@(w) (w .* mag_sqr_S0(w)).^2 / V_arr(w,ai), -Inf, Inf, 'ArrayValued', true));

                        if (scheme == "PPC" || scheme == "NPC")
                            %%% CWE PPC AND NPC %%%
                            % Define derivative expressions for CRLBs.
                            
                            % Compute decorrelated derivative expressions for alpha.
                            dsdR_alpha = dt*(pagemtimes(pagetranspose(real(dsalpha)),hdR_matrix) - pagemtimes(pagetranspose(imag(dsalpha)),hdI_matrix));
                            dsdI_alpha = dt*(pagemtimes(pagetranspose(imag(dsalpha)),hdR_matrix) + pagemtimes(pagetranspose(real(dsalpha)),hdI_matrix));
    
                            % Compute decorrelated derivative expressions for t0.
                            dsdR_t0 = dt*(pagemtimes(pagetranspose(real(dst0)),hdR_matrix) - pagemtimes(pagetranspose(imag(dst0)),hdI_matrix));
                            dsdI_t0 = dt*(pagemtimes(pagetranspose(imag(dst0)),hdR_matrix) + pagemtimes(pagetranspose(real(dst0)),hdI_matrix));
                            dsdR_t0_tr = pagetranspose(dsdR_t0);
                            dsdI_t0_tr = pagetranspose(dsdI_t0);
    
                            % Compute fisher information matrix (FIM).
                            fisher_matrix = dt*dt*[(pagemtimes(pagemtimes(dsdR_alpha,QnR_matrix),pagetranspose(dsdR_alpha)) + pagemtimes(pagemtimes(dsdI_alpha,QnI_matrix),pagetranspose(dsdI_alpha)))...
                                        (pagemtimes(pagemtimes(dsdR_alpha,QnR_matrix),dsdR_t0_tr) + pagemtimes(pagemtimes(dsdI_alpha,QnI_matrix),dsdI_t0_tr));...
                                        (pagemtimes(pagemtimes(dsdR_alpha,QnR_matrix),dsdR_t0_tr) + pagemtimes(pagemtimes(dsdI_alpha,QnI_matrix),dsdI_t0_tr))...
                                        (pagemtimes(pagemtimes(dsdR_t0,QnR_matrix),dsdR_t0_tr) + pagemtimes(pagemtimes(dsdI_t0,QnI_matrix),dsdI_t0_tr))];

                            % Compute inverse of FIM, should be diagonal matrix,
                            % diagonal terms are CRLBs.
                            inv_zz = pageinv(fisher_matrix);
                            % Compute CRLB for alpha.
                            crlb(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:) = inv_zz(1,1,:);
                            % Compute CRLB for t0.
                            crlb(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:) = inv_zz(2,2,:);
                        
                        % Compute AC-CWE CRLBs for full phase compensation
                        elseif (scheme == "MPC" || scheme == "EPC") 
                            % Compute FIM.
                            fisher_matrix = dt*dt*[pagemtimes(pagemtimes(pagetranspose(dsalpha),Qn_matrix),dsalpha), pagemtimes(pagemtimes(pagetranspose(dst0),Qn_matrix),dsalpha);...
                                pagemtimes(pagemtimes(pagetranspose(dst0),Qn_matrix),dsalpha), pagemtimes(pagemtimes(pagetranspose(dst0),Qn_matrix),dst0)];
                            % Compute inverse of FIM.
                            inv_zz = pageinv(fisher_matrix);
                            % Compute CRLB for alpha.
                            crlb(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:) = inv_zz(1,1,:);
                            % Compute CRLB for t0.
                            crlb(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:) = inv_zz(2,2,:);
                        end
                    end
                    disp('')
                end
            end
        end
    end
end

% Stop run timer.
runtime = toc(loopTic);
% Display total run time.
disp(experiment + " took " + floor(runtime/60) + " minutes " + mod(runtime,60) + " seconds")

%% Compute averages across deployments.
if nDeployments == 1
    avg_dep_var = empirical_var;
    avg_dep_mse = empirical_mse;
    avg_dep_bias = empirical_bias;
    avg_dep_crlb = crlb;
    avg_dep_my_var = my_var;
else
    avg_dep_var = mean(empirical_var,length(size(empirical_var)));
    avg_dep_mse = mean(empirical_mse,length(size(empirical_mse)));
    avg_dep_bias = mean(empirical_bias,length(size(empirical_bias)));
    avg_dep_crlb = mean(crlb,length(size(crlb)));
    avg_dep_my_var = mean(my_var,length(size(crlb)));
end

%% Save experiment results.
if ~any(experiment == experiment_list)
    error("Unknown experiment specified. Results not saved!")
else
    save(experiment + "_results.mat", "avg_dep_var", "avg_dep_mse", "avg_dep_bias", "avg_dep_crlb",...
        "agent_db_values", "selected_schemes", "dropout_vals", "sensor_vals", "channel_db_values",...
        "experiment","sensor_dimension")
end

%% Plotting
% params_latex = ["\alpha","t_0"];
% params_text = ["alpha","t0"];
% color_vec = ["#A2142F", "#0072BD", "#333333"];
% 
% home_dir = char(java.lang.System.getProperty('user.home'));
% folder_path = home_dir + "\Desktop\Updated Final AirComp Results\";
% 
% plots = true;
% save_figs = true;
% 
% if plots
%     % Linear plots
%     if experiment ~= "cwe_joint_grid"
%         for agent_db_idx = 1:length(agent_db_values)
%             % Create save directory if it does not exist.
%             save_folder = folder_path + experiment + "_" + string(agent_db_values(agent_db_idx)) + "_db";
%             if ~exist(save_folder, 'dir')
%                 mkdir(save_folder);
%             end
% 
%             for param_idx = 1:2
%                 % Collect all plot values for y-axis scaling.
%                 all_vals = [avg_dep_var(:,agent_db_idx,:,param_idx,:,1:sensor_dimension);...
%                             avg_dep_mse(:,agent_db_idx,:,param_idx,:,1:sensor_dimension);...
%                             avg_dep_crlb(:,agent_db_idx,:,param_idx,:,1:sensor_dimension)];
%                 for scheme_idx = 1:length(selected_schemes)
%                     scheme = selected_schemes(scheme_idx);
%                     fig1 = figure;
%                     set(fig1,'Position',[100,100,450,450])
%                     ax = gca;
%                     hold on
% 
%                     % Plot pseudoplots for legend symbols.
%                     if experiment == "random_dropout"
%                         for dropout_idx = 1:length(dropout_vals)
%                             plot(nan,nan,'color',color_vec(dropout_idx),'LineWidth',2);
%                         end
%                     else
%                         for sensor_idx = 1:length(sensor_vals) 
%                             plot(nan,nan,'color',color_vec(sensor_idx),'LineWidth',2);
%                         end
%                     end
%                     plot(nan,nan,'^','color','black');
%                     plot(nan,nan,'x','color','black');
%                     plot(nan,nan,'o','color','black');
% 
%                     x_axis_series = channel_db_values(scheme_idx,:,agent_db_idx);
% 
%                     if experiment == "random_dropout"
%                         count_vals = dropout_vals;
%                     else
%                         count_vals = sensor_vals;
%                     end
% 
%                     if scheme == "NPC"
%                         count_vals = count_vals(1);
%                     end
% 
%                     for count_idx = 1:length(count_vals)
%                         plot(x_axis_series,(squeeze(avg_dep_var(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-^','color',color_vec(count_idx),'LineWidth', 1)
%                         plot(x_axis_series,(squeeze(avg_dep_mse(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-x','color',color_vec(count_idx),'LineWidth', 1)
%                         plot(x_axis_series,(squeeze(avg_dep_crlb(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'--o','color',color_vec(count_idx),'LineWidth', 1)
%                     end
% 
%                     xlabel('Channel SNR (dB)')
%                     ylabel(" ")
% 
%                     title('$$\hat{'+params_latex(param_idx)+'}$$','Interpreter','latex')
% 
%                     set(ax, 'FontSize', 15);
%                     set(ax, 'YScale', 'log')
% 
%                     if scheme == "MPC"
%                         if experiment == "random_dropout"
%                             legend(["Drop = " + dropout_vals,"VAR","MSE","CRLB"], 'Location', 'best')
%                         else
%                             legend(["S = " + sensor_vals,"VAR","MSE","CRLB"], 'Location', 'best')
%                         end
%                     end
% 
%                     % Add box around plot axes.
%                     box(ax, 'on');
%                     grid on
% 
%                     y_lower = 10^(-0.25)*min(all_vals,[],"all");
%                     y_upper = 1.05*max(all_vals,[],"all");
%                     ylim([y_lower,y_upper]);
%                     y_ticks = power_10_range(y_lower,y_upper);
% 
%                     % Don't plot for certain combinations due to axis and box
%                     % overlap
%                     if param_idx == 1
%                             y_ticks = y_ticks(2:end);
%                     end
% 
%                     ax.YTick = y_ticks;
%                     ax.GridColor = [0 0 0];
%                     ax.GridAlpha = 0.5;
%                     ax.LineWidth = 1;
% 
%                     % Save the figure.
%                     if save_figs
%                         exportgraphics(fig1, fullfile(save_folder, sprintf(experiment + "_" + string(agent_db_values(agent_db_idx)) + "_db_" + params_text(param_idx) + "_" + selected_schemes(scheme_idx) + ".png")), 'Resolution', 300);
%                     end
%                     % close(fig1)
%                 end
%             end
%         end
%     % Surface plots
%     else
%         % Create save directory if it does not exist.
%         save_folder = folder_path + experiment + "_" + string(agent_db_values(end)) + "_db";
%         if ~exist(save_folder, 'dir')
%             mkdir(save_folder);
%         end
% 
%         % Toggle if you want to convert meshgrid values to log (you should).
%         convert_to_log = true;
% 
%         for param_idx = 1:2
%             % Collect all plot values for y-axis scaling.
%             all_vals = [avg_dep_var(:,:,:,param_idx,:,1:sensor_dimension);...
%                         avg_dep_mse(:,:,:,param_idx,:,1:sensor_dimension);...
%                         avg_dep_crlb(:,:,:,param_idx,:,1:sensor_dimension)];
%             if convert_to_log
%                 all_vals = log10(all_vals);
%             end
% 
%             for scheme_idx = 1:length(selected_schemes)
%                 fig1 = figure;
%                 set(fig1,'Position',[100,100,675,475])
%                 ax = gca;
% 
%                 count_vals = sensor_vals;
% 
%                 for count_idx = 1:length(count_vals)
%                     z_vals = squeeze(avg_dep_var(:,:,:,param_idx,scheme_idx,count_idx,1));
%                     if convert_to_log
%                         z_vals = log10(z_vals);
%                     end
%                     surf(agent_db_values, channel_db_values(1,:,1), z_vals)
%                 end
% 
%                 cb = colorbar;
%                 colormap(turbo)
%                 % Change view angle to better see both x and y dimensions
%                 view(150,20)
%                 xl = xlabel('Channel SNR (dB)', 'FontWeight', 'bold');
%                 yl = ylabel('Sensor SNR (dB)', 'FontWeight', 'bold');
% 
% 
%                 title('$$\hat{' + params_latex(param_idx) + '}$$', 'Interpreter','latex')
%                 set(ax, 'FontSize', 15);
% 
%                 zlabel('$$ \mathbf{\log_{10} ((\mathrm{Var} ( \hat{'+params_latex(param_idx)+'} ))}$$','Interpreter','latex', 'FontSize', 20)
% 
%                 % Add box around plot axes.
%                 box(ax, 'on');
%                 grid on
% 
%                 if convert_to_log
%                     z_lower = 1.05*min(all_vals,[],"all");
%                 else
%                     set(ax, 'ZScale', 'log')
%                     z_lower = 10^(-0.25)*min(all_vals,[],"all");
%                 end
% 
%                 z_upper = 1.05*max(all_vals,[],"all");
% 
%                 zlim([z_lower,z_upper]);
%                 clim([z_lower,z_upper])
% 
%                 if ~convert_to_log
%                     z_ticks = power_10_range(z_lower,z_upper);
%                     ax.ZTick = z_ticks;
%                 end
% 
%                 ax.GridColor = [0 0 0];
%                 ax.GridAlpha = 1;
%                 ax.LineWidth = 1;
% 
%                 xpos = get(ax, 'Xlabel');
%                 xpos = xpos.Position;
%                 xpos(1) = xpos(1) + 2;
%                 xpos(2) = xpos(2) - 2;
% 
%                 xl.Position = xpos;
%                 xl.Rotation = 10;
% 
%                 ypos = get(ax, 'Ylabel');
%                 ypos = ypos.Position;
%                 ypos(1) = ypos(1) - 2;
%                 ypos(2) = ypos(2) + 2.5;
% 
%                 % if scheme_idx == 1
%                 %     ypos(1) = ypos(1) - 5.5;
%                 %     ypos(2) = ypos(2) - 2.5;
%                 % else
%                 %     ypos(1) = ypos(1) - 4.5;
%                 %     ypos(2) = ypos(2) - 2.5;
%                 % end
% 
%                 yl.Position = ypos;
%                 yl.Rotation = -30;
% 
%                 axis normal
% 
%                 % Save the figure.
%                 if save_figs
%                     exportgraphics(fig1, fullfile(save_folder, sprintf(experiment + "_" + string(agent_db_values(end)) + "_db_" + params_text(param_idx) + "_" + selected_schemes(scheme_idx) + ".png")), 'Resolution', 300);
%                 end
%                 % close(fig1)
%             end
%         end
%     end
% end