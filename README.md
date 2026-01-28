This repository contains the MATLAB code used to generate the numerical results in my paper titled "Over-the-Air Computation on 
Network Edge for Collaborative Estimation". Plaintext versions of the .m files can be found in the .dat Files folder.

# How to generate your own results:
- Inside alpha_t0.m, change the "experiment" variable to run the desired 
experiment corresponding to the paper. All other parameters are tailored to 
match the exact implementation used in the paper. The individual experiment 
plots will be generated and saved to the path specified in "folder_path".
- Run the corresponding "plot_..." script to plot the figures for a given experiment.
- The "plot_..." scripts will specify what result files are required to generate the figures.

If you encounter any issues with the script, please feel free to reach out to me at vcvhuynh (at) ucdavis (dot) edu.

# Known Errata
A list of known errata has been compiled since initial submission of the manuscript for review (9/7/2025). The results generated after correcting the errata still align with the claims presented in original manuscript. We advise the user to compare the version date of the repository with the fix dates of the errata (where applicable) to understand which issues may still be persisting in the code.

## 1/27/2026
- The variable names num_accum_compare and denom_accum_compare should be changed to num_accum and denom_accum

## 9/30/2025  

- **FIXED 1/6/2026**: When running the code with `experiment = random_dropout"`, the CRLB is not computed. Around line **487**, update the condition:
```matlab
% Before
elseif exp_split(1) == "cwe"

% After
elseif exp_split(1) == "cwe" || experiment == "random_dropout"
```

- **FIXED 1/6/2026**: There is an inconsistency between the results for `cwe_joint` and `random_dropout` because of the way random number generation is handled in MATLAB. For `Drop = 0`, the curves should line up. 
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

- **FIXED 1/6/2026**: Around line **109**, add the following logic after creating `all_tpi`:
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

- **FIXED 1/6/2026**: There is a missing factor of `dt`. Around line **212**, update the expression for `gamma_n`:
```matlab
% Before
gamma_n = P_s * E_mi_sqr * E_mag_g_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));

% After
gamma_n = dt * P_s * E_mi_sqr * E_mag_g_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
```
