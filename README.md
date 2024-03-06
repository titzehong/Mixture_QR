# EPA QR Mixture
This repository contains code to run the mixture EPA model. 

## Data Setup
- Covariates (X) can be univariate or multivariate however responses (Y) have to be univariate
- Model autoamtically detects if X are multivariate and does the appropriate projection as described.

## To run just QR
```
from Single_Var_QR import SingleQRSampler_T_4_block

quantile_dist = 'norm'
prior_on_t = False

# Choose no. of chains
n_chains = 2
n_steps = 20000

# Set Grid of tau values
tau_grid_expanded = np.array([round(-0.01 + 0.01 * i,2) for i in range(103)])
tau_grid = np.array([round(0.01 + 0.01 * i,2) for i in range(99)])
knot_points_grid = np.arange(0.1,1,0.1)

data_size = n_data
tau_upper_tail = 1-1/(2*data_size)
tau_lower_tail = 1/(2*data_size)
lower_seq = np.flip(geometric_seq(0.01, tau_lower_tail, 0.005, upper=False))
upper_seq = geometric_seq(0.99, tau_upper_tail, 0.005)

tau_grid = np.array([round(0.01 + 0.01 * i,2) for i in range(99)])
tau_grid = np.concatenate([lower_seq, tau_grid, upper_seq])

tau_grid_expanded = np.concatenate([np.array([-0.01,0]),
                                    tau_grid,
                                    np.array([1,1.01])])

# Run Sampler
sampler_collecter_4blockt = SingleQRSampler_T_4_block(y_train,
                                    X_train,          
                                    C_1 = 0.3,
                                    lambda_step_size_1 = 3,
                                    alpha_step_size_1 = 0.4,
                                    a_target_1 = 0.228,
                                    C_2 = 0.3,
                                    lambda_step_size_2 = 3,
                                    alpha_step_size_2 = 0.4,
                                    a_target_2 = 0.228,
                                    C_3 = 0.5,
                                    lambda_step_size_3 = 3,
                                    alpha_step_size_3 = 0.4,
                                    a_target_3 = 0.228,
                                    C_4 = 0.3,
                                    lambda_step_size_4 = 3,
                                    alpha_step_size_4 = 0.4,
                                    a_target_4 = 0.228,
                                    tau_grid_expanded = tau_grid_expanded,
                                    tau_grid = tau_grid,     
                                    knot_points_grid = knot_points_grid,
                                    am_lamb_block1_init = (2.38**2)/(9),
                                    am_lamb_block2_init = (2.38**2)/(9),
                                    am_lamb_block3_init = (2.38**2)/4,
                                    am_lamb_block4_init = (2.38**2)/(9*2+4),
                                    alpha_kappa = 0.1,
                                    beta_kappa = 0.1,
                                    eps_1 = 0,
                                    eps_2 = 0,
                                    base_quantile_mean=0.0,
                                    base_quantile_sd=1.0,
                                    base_quantile_v=1.0,
                                    base_quantile_dist=quantile_dist,
                                    prior_on_t=prior_on_t,
                                    splice=True) 
```

## To run just EPA Regression

## To run EPA - QR 
