This repository contains the MATLAB code used to generate the numerical results in my paper titled "Over-the-Air Computation on 
Network Edge for Collaborative Estimation". Plaintext versions of the .m files can be found in the .dat Files folder.

# Known Errata
A list of know errata is being compiled while the review process is underway. Please note that the errata do not have drastic effects on the subsequent results for the paper. 
We have verified that results regenerated after correcting these errata still support the trends and conclusions presented in our original manuscript.

## 9/30/2025  
When running the code with `experiment = random_dropout"`, the CRLB is not computed. Around line **487**, update the condition:

```matlab
% Before
elseif exp_split(1) == "cwe"

% After
elseif exp_split(1) == "cwe" || experiment == "random_dropout"
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
