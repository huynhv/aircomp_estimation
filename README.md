This repository contains the MATLAB code used to generate the numerical results in my paper titled "Over-the-Air Computation on 
Network Edge for Collaborative Estimation".

# How to generate your own results:
- Inside alpha_t0.m, change the "experiment" variable to run the desired 
experiment corresponding to the paper. All other parameters are tailored to 
match the exact implementation used in the paper. The individual experiment 
plots will be generated and saved to the path specified in "folder_path".
- To generate the overlayed results for AC-NAE and AC-CWE, run the 
plot_nae_cwe_same_plot.m file after generating the results. The plots will be 
generated and saved to the path specified in "folder_path".
