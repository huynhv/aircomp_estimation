This repository contains the MATLAB code used to generate the numerical results in my paper titled "Over-the-Air Computation on 
Network Edge for Collaborative Estimation". Plaintext versions of the .m files can be found in the .dat Files folder.

# Known Errata
A list of know errata is being compiled while the review process is underway. We have verified that regenerated results after correcting these errata still support the trends and conclusions presented in our original manuscript.

## 9/30/2025  
- When running the code with `experiment = random_dropout"`, the CRLB is not computed. Around line **487**, update the condition:
```matlab
% Before
elseif exp_split(1) == "cwe"

% After
elseif exp_split(1) == "cwe" || experiment == "random_dropout"
```

- There is an inconsistency between the results for `cwe_joint` and `random_dropout` because of the way random number generation is handled in MATLAB. For `Drop = 0`, the curves should line up. 
Around line **28**, modify the code inside the if statement:
```matlab
% Before
if experiment == "random_dropout"
    dropout_vals = [0,2,4];
    agent_db_values = [0];
    sensor_dimension = length(dropout_vals);
    % Here we only consider maximum number of sensors and drop from there.
    sensor_vals = [10];
    
    which_dropout = zeros(nDeployments,max(dropout_vals));
    for idx = 1:nDeployments
        which_dropout(idx,:) = randperm(max(sensor_vals),max(dropout_vals));
    end

% After
if experiment == "random_dropout"
    dropout_vals = [0,2,4];
    agent_db_values = [0];
    sensor_dimension = length(dropout_vals);
    % Here we only consider maximum number of sensors and drop from there.
    sensor_vals = [10];
```

Around line **109**, add the following logic after creating `all_tpi`:
```matlab
% Before
if experiment == "nae_joint_imperfect_tpi" || experiment == "cwe_joint_imperfect_tpi"
    all_tpi = unifrnd(min_tpi,max_tpi,1,max(sensor_vals),1,nDeployments);
else
    all_tpi = zeros(1,max(sensor_vals),1,nDeployments);
end

% After 
if experiment == "nae_joint_imperfect_tpi" || experiment == "cwe_joint_imperfect_tpi"
    all_tpi = unifrnd(min_tpi,max_tpi,1,max(sensor_vals),1,nDeployments);
else
    all_tpi = zeros(1,max(sensor_vals),1,nDeployments);
end

if experiment == "random_dropout"
    which_dropout = zeros(nDeployments,max(dropout_vals));
    for idx = 1:nDeployments
        which_dropout(idx,:) = randperm(max(sensor_vals),max(dropout_vals));
    end
end
``` 

- There is a missing factor of `dt`. Around line **212**, update the expression for `gamma_n`:
```matlab
% Before
gamma_n = P_s * E_mi_sqr * E_mag_g_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));

% After
gamma_n = dt * P_s * E_mi_sqr * E_mag_g_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
```

# How to generate your own results:
- Inside alpha_t0.m, change the "experiment" variable to run the desired 
experiment corresponding to the paper. All other parameters are tailored to 
match the exact implementation used in the paper. The individual experiment 
plots will be generated and saved to the path specified in "folder_path".
- To generate the overlayed results for AC-NAE and AC-CWE, run the 
plot_nae_cwe_same_plot.m file after generating the results. The plots will be 
generated and saved to the path specified in "folder_path".
- Run nae_variance.m to generate the ablation study plot for the upper bound of the NAE variance.
- Run plot_perfect_and_imperfect_tpi.m to generate the overlayed ablation study plots for imperfect clock synchronization.
