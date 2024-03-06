import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import numba

from sample_processing import *
from gibbs_sampler import *


dataset_choice = 2

np.random.seed(1)
noise = 'norm'
n_points_per_clust = 150
Xs = np.concatenate([np.random.uniform(-1,0,n_points_per_clust),
                    np.random.uniform(0,1,n_points_per_clust)])

inds = np.array(n_points_per_clust*[0] + n_points_per_clust*[1])

betas = [np.array([-1,4]),
         np.array([1,-4])]

#betas = [np.array([1,-4]),
#         np.array([1,-4])]

if noise == 'norm':
    Ys = np.concatenate([betas[0][0] + betas[0][1]*Xs[0:n_points_per_clust] + np.random.normal(0,2,n_points_per_clust),
                     betas[1][0] + betas[1][1]*Xs[n_points_per_clust:]+ np.random.normal(0,2,n_points_per_clust)])
else: # t
    Ys = np.concatenate([betas[0][0] + betas[0][1]*Xs[0:n_points_per_clust] + np.random.standard_t(4,n_points_per_clust),
                     betas[1][0] + betas[1][1]*Xs[n_points_per_clust:]+ np.random.standard_t(4,n_points_per_clust)])


from sklearn.model_selection import train_test_split
from scipy.stats import norm

X_train, X_test,\
    y_train, y_test,\
    inds_train, inds_test = train_test_split(Xs, Ys,
                                            inds,
                                            stratify=inds,
                                            test_size=0.25, random_state=42)


y_train = y_train
X_train = X_train
n_data = len(y_train)


from Single_Var_QR import SingleQRSampler_T_4_block

quantile_dist = 't'

# Choose no. of chains
n_chains = 2

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
sampler_collecter_4blockt = [SingleQRSampler_T_4_block(y_train,
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
                                    prior_on_t=True,
                                    splice=True) for _ in range(n_chains)]

chain_outputs = [sampler_c.sample(n_steps=25000) for sampler_c in sampler_collecter_4blockt]

sampler_collecter = sampler_collecter_4blockt