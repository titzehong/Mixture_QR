import numpy as np
import matplotlib.pyplot as plt
import numba 
from numba import prange
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal, gamma, multivariate_t, logistic
from scipy.stats import t as student_t
import time


from Single_Var_QR_utils import *
from GP_Approx import *

#########################
#### 4 Block Sampler ####
#########################

class SingleQRSampler_T_4_block:

    def __init__(self, 
                y_vals_true,
                x_vals,          
                C_1 = 0.5,
                lambda_step_size_1 = 3,
                alpha_step_size_1 = 0.45,
                a_target_1 = 0.4,
                C_2 = 0.5,
                lambda_step_size_2 = 3,
                alpha_step_size_2 = 0.45,
                a_target_2 = 0.4,
                C_3 = 0.5,
                lambda_step_size_3 = 3,
                alpha_step_size_3 = 0.45,
                a_target_3 = 0.4,
                C_4 = 0.5,
                lambda_step_size_4 = 3,
                alpha_step_size_4 = 0.45,
                a_target_4 = 0.4,
                tau_grid_expanded = np.arange(-0.01,1.02,0.01),
                tau_grid = np.arange(0.01,1.0,0.01),     
                knot_points_grid = np.arange(0.1,1,0.1),
                am_lamb_block1_init = 0.1,
                am_lamb_block2_init = 0.5,
                am_lamb_block3_init = 0.5,
                am_lamb_block4_init = 0.5,
                alpha_kappa = 5,
                beta_kappa = 1/3,
                eps_1 = 0,
                eps_2 = 0,
                eps_3 = 0,
                eps_4 = 0,
                base_quantile_mean=0.0,
                base_quantile_sd=1.0,
                base_quantile_v=1,
                base_quantile_dist='norm',
                prior_on_t=False,
                splice=True,
                verbose=False,
                track_block=True):
        
        # 
        self.track_block = track_block

        # Whether to splice in LL evaluation
        self.splice=splice

        # Whether to use a prior on t
        self.prior_on_t = prior_on_t

        # Where multivariate X
        self.multivariate_x = False
        if len(x_vals.shape) > 1:
            if x_vals.shape[1] > 1:
                self.multivariate_x = True

        if self.multivariate_x:
            self.n_covar = x_vals.shape[1]

        if self.multivariate_x:
            print('Multivariate model') 
        else:
            print("Univariate Model")       

        # Block Sampling proportions
        self.m = len(knot_points_grid)
        if not self.prior_on_t:
            self.param_sizes = np.array([self.m,self.m,4,self.m+self.m+4])
            
            if self.multivariate_x:
                self.param_sizes = np.array([self.m,self.m,4 + self.n_covar,self.m+self.m+4 + self.n_covar])

        else:
            self.param_sizes = np.array([self.m,self.m,5,self.m+self.m+5])

            if self.multivariate_x:
                self.param_sizes = np.array([self.m,self.m,5 + self.n_covar,self.m+self.m+5 + self.n_covar])

        self.param_sizes_prop = self.param_sizes/self.param_sizes.sum()

        # AM Sampler hyper params Block 1 (First GP)
        self.C_1 = C_1
        self.lambda_step_size_1 =lambda_step_size_1
        self.alpha_step_size_1 = alpha_step_size_1
        self.a_target_1 = a_target_1
        self.am_lamb_block1_init =am_lamb_block1_init
        self.eps_1 = eps_1
        self.alpha_kappa = alpha_kappa
        self.beta_kappa = beta_kappa

        # AM Sampler hyper params Block 2 (Second GP)
        self.C_2 =C_2
        self.lambda_step_size_2 =lambda_step_size_2
        self.alpha_step_size_2 = alpha_step_size_2
        self.a_target_2 =a_target_2
        self.am_lamb_block2_init = am_lamb_block2_init
        self.eps_2 = eps_2

        # AM Sampler hyper params Block 3 (Remaining Params)
        self.C_3 =C_3
        self.lambda_step_size_3 =lambda_step_size_3
        self.alpha_step_size_3 = alpha_step_size_3
        self.a_target_3 =a_target_3
        self.am_lamb_block3_init = am_lamb_block3_init
        self.eps_3= eps_3

        # AM Sampler hyper params Block 4 (All params)
        self.C_4 =C_4
        self.lambda_step_size_4 =lambda_step_size_4
        self.alpha_step_size_4 = alpha_step_size_4
        self.a_target_4 =a_target_4
        self.am_lamb_block4_init = am_lamb_block4_init
        self.eps_4= eps_4

        # Tau grid 
        self.tau_grid_expanded =tau_grid_expanded
        self.tau_grid = tau_grid
        self.knot_points_grid =knot_points_grid

        # Quantile function parameters
        self.base_quantile_mean = base_quantile_mean
        self.base_quantile_sd = base_quantile_sd
        self.base_quantile_v = base_quantile_v
        self.base_quantile_dist = base_quantile_dist

        # Storage variables
        self.w1_knot_store = []
        self.w2_knot_store = []
        self.w1_approx_store = []
        self.w2_approx_store = []
        self.w_approx_store = []

        self.kappa_store = []
        self.lambda_store = []
        self.rho_store = []

        self.sigma_1_store = []
        self.sigma_2_store = []
        self.mu_store = []
        self.gamma_store = []
        self.v_store = []

        self.x_alpha_store = []

        # Data
        self.y_vals_true = y_vals_true
        self.x_vals = x_vals

        self.block_1_accept_cnts = 0
        self.block_2_accept_cnts = 0
        self.block_3_accept_cnts = 0
        self.block_4_accept_cnts = 0

        # Store stuff for logging
        self.ll_check_blk1 = []
        self.ll_check_blk2 = []
        self.ll_check_blk3 = []
        self.ll_check_blk4 = []

        self.prop_check1 = []
        self.prop_check2 = []
        self.prop_check3 = []
        self.prop_check4 = []

        self.a_check_1 = []
        self.cov_store_2 = []
        self.mu_block_store_2 =[]
        self.am_params_store_2 = []
        self.a_check_2 = []
        self.blk2_check = []
        self.log_score = []
        self.a_check_3 = []
        self.cov_store_3 = []
        self.blk3_check = []
        self.a_check_4 = []
        self.cov_store_4 = []
        self.blk4_check = []
        
        # Set up priors for everything
        self.mu_mean = 0
        self.mu_scale = 1

        self.gamma_mean = 0
        self.gamma_scale = 1

        self.sigma_sq_1_a = 2
        self.sigma_sq_1_b = 2

        self.sigma_sq_2_a = 2
        self.sigma_sq_2_b = 2

        self.verbose = verbose


    def sample(self, n_steps=20000, n_burn_in=5000):

        # Start time
        s = time.time()

        self.n_steps = n_steps

        # Initialize Step sizes
        step_sizes_1 = self.C_1/(np.arange(1,n_steps+10)**self.alpha_step_size_1)
        step_sizes_2 = self.C_2/(np.arange(1,n_steps+10)**self.alpha_step_size_2)
        step_sizes_3 = self.C_3/(np.arange(1,n_steps+10)**self.alpha_step_size_3)
        step_sizes_4 = self.C_4/(np.arange(1,n_steps+10)**self.alpha_step_size_4)


        # Calc covariance matrix 
        tau_grid_expanded = self.tau_grid_expanded
        tau_grid = self.tau_grid  
        knot_points_grid = self.knot_points_grid
        m = len(knot_points_grid)
        

        # Get Lambda PDF
        h=0.1
        lambd_collect = calc_lambd_grid_uncorr(knot_points_grid.reshape(-1,1),
                                            h=h,
                                            rho_lambd_init=0.99)
        lambd_grid = np.array(lambd_collect)
        grid_pdf_vals = lamb_pdf(lambd_grid)
        lambda_grid_log_prob = np.log(grid_pdf_vals)


        #### Initialize Model Parameters
        ## GP Related hyperparameters
        kappa_current = np.nan
        rho_current = 0.1
        lambd_current = 4
        alpha_kappa = 3
        beta_kappa = 1/3


        ## Regression related parametrs
        mu_current = 0
        if not self.multivariate_x:
            gamma_current = 0
        else:
            gamma_current = np.array([student_t.rvs(df=1,loc=0,scale=1) for _ in range(self.n_covar)])  # 4 independent t distribution
            
        sigma_1_current = 1
        sigma_2_current = 1
        v_current = 1  # Only used if logistic prior on t ppf used
        v_prop = 1 # Define first in case prior not used
        
        ## Multivariate projection related parameted (if needed)
        if self.multivariate_x:
            x_alpha_current = multivariate_t.rvs(loc = np.zeros(self.n_covar),
                                                shape=np.eye(self.n_covar),
                                                df=1)  # Draw from t_1 (0,1) distribution
        else:
            x_alpha_current = None
        #### W samples 
        # calc covariance matrix
        cov_mat_knots_current = covariance_matrix_gp_uncorr(knot_points_grid.reshape(-1,1),
                                          lambd=lambd_current)
        # Precompute Matrices
        cov_matrices_G, A_matrices_G = precompute_approx_uncorr(tau_grid_expanded.reshape(-1,1),
                                                           knot_points_grid.reshape(-1,1),
                                                           lambda_grid=lambd_grid)


        # Generate some data for the current knot points (randomly)
        w1_knot_points_current = multivariate_t.rvs(loc=np.zeros(m),
                              shape=cov_mat_knots_current*(beta_kappa/alpha_kappa),
                              df=2*alpha_kappa)
        
        w2_knot_points_current = multivariate_t.rvs(loc=np.zeros(m),
                            shape=cov_mat_knots_current*(beta_kappa/alpha_kappa),
                            df=2*alpha_kappa)



        # Generate sample of GP approx
        w1_approx_current, lp_w1_current = calc_mixture_knot_approx_marginalized(w1_knot_points_current,
                                                                                a_kappa=alpha_kappa,
                                                                                b_kappa=beta_kappa,
                                                                                tau_grid=tau_grid_expanded,
                                                                                A_g_matrices=A_matrices_G,
                                                                                cov_mat_knot_store=cov_matrices_G,
                                                                                lambda_grid_log_prob=lambda_grid_log_prob)
        
        w2_approx_current, lp_w2_current = calc_mixture_knot_approx_marginalized(w2_knot_points_current,
                                                                                        a_kappa=alpha_kappa,
                                                                                        b_kappa=beta_kappa,
                                                                                        tau_grid=tau_grid_expanded,
                                                                                        A_g_matrices=A_matrices_G,
                                                                                        cov_mat_knot_store=cov_matrices_G,
                                                                                        lambda_grid_log_prob=lambda_grid_log_prob)

        ### initialise adaptive metropolis

        # Block 1
        am_lamb_block1 = self.am_lamb_block1_init
        log_am_lamb_block1 = np.log(am_lamb_block1)
        am_cov_block1 = block_diag(cov_mat_knots_current)

        mu_block1 = w1_knot_points_current
            
        # Block 2
        am_lamb_block2 = self.am_lamb_block2_init
        log_am_lamb_block2 = np.log(am_lamb_block2)
        am_cov_block2 = block_diag(cov_mat_knots_current)

        mu_block2 = w2_knot_points_current

        # Block 3
        am_lamb_block3 = self.am_lamb_block3_init
        log_am_lamb_block3 = np.log(am_lamb_block3)
        if not self.prior_on_t:
            if not self.multivariate_x:
                am_cov_block3 = np.diag([np.sqrt(5),np.sqrt(5),1,1])
                mu_block3= np.concatenate([np.array([mu_current]),
                                            np.array([gamma_current]),
                                            np.array([np.log(sigma_1_current),
                                                        np.log(sigma_2_current)])])
            
            else:
                am_cov_block3 = np.diag([np.sqrt(5)]+self.n_covar*[np.sqrt(5)]+[1,1]+self.n_covar*[1])
                mu_block3= np.concatenate([np.array([mu_current]),
                                gamma_current,
                                np.array([np.log(sigma_1_current),
                                            np.log(sigma_2_current)]),
                                x_alpha_current])


        else:
            if not self.multivariate_x:
                am_cov_block3 = np.diag([np.sqrt(5),np.sqrt(5),1,1, 1])
                mu_block3= np.concatenate([np.array([mu_current]),
                                            np.array([gamma_current]),
                                            np.array([np.log(sigma_1_current),
                                                        np.log(sigma_2_current)]),
                                            np.array([np.log(v_current)])])

            else:
                am_cov_block3 = np.diag([np.sqrt(5)] + self.n_covar*[np.sqrt(5)] + [1,1,1]+self.n_covar*[1])
                mu_block3= np.concatenate([np.array([mu_current]),
                                        gamma_current,
                                        np.array([np.log(sigma_1_current),
                                                    np.log(sigma_2_current)]),
                                        np.array([np.log(v_current)]),
                                        x_alpha_current])

        
        # Block 4
        am_lamb_block4 = self.am_lamb_block4_init
        log_am_lamb_block4 = np.log(am_lamb_block4)
        am_cov_block4 = block_diag(cov_mat_knots_current,cov_mat_knots_current,am_cov_block3)  # 9 + 9 + 4 dim diagonal covariance matrix
        
        if not self.prior_on_t:
            if not self.multivariate_x:
                mu_block4= np.concatenate([w1_knot_points_current,
                                        w2_knot_points_current,
                                        np.array([mu_current]),
                                            np.array([gamma_current]),
                                            np.array([np.log(sigma_1_current),
                                                        np.log(sigma_2_current)])])
            
            else:
                mu_block4= np.concatenate([w1_knot_points_current,
                            w2_knot_points_current,
                            np.array([mu_current]),
                                gamma_current,
                                np.array([np.log(sigma_1_current),
                                            np.log(sigma_2_current)]),
                            x_alpha_current])

        else:
            if not self.multivariate_x:
                mu_block4= np.concatenate([w1_knot_points_current,
                                w2_knot_points_current,
                                np.array([mu_current]),
                                    np.array([gamma_current]),
                                    np.array([np.log(sigma_1_current),
                                                np.log(sigma_2_current)]),
                                    np.array([np.log(v_current)])])
            else:
                mu_block4= np.concatenate([w1_knot_points_current,
                                w2_knot_points_current,
                                np.array([mu_current]),
                                    gamma_current,
                                    np.array([np.log(sigma_1_current),
                                                np.log(sigma_2_current)]),
                                    np.array([np.log(v_current)]),
                                    x_alpha_current])


        # Set up initial state
    
        update_block_1 = w1_knot_points_current

        update_block_2 = w2_knot_points_current

        if not self.prior_on_t:
            if not self.multivariate_x:
                update_block_3 = np.concatenate([np.array([mu_current]),
                                        np.array([gamma_current]),
                                        np.array([np.log(sigma_1_current),
                                                    np.log(sigma_2_current)])])
            
            else:
                update_block_3 = np.concatenate([np.array([mu_current]),
                                        gamma_current,
                                        np.array([np.log(sigma_1_current),
                                                    np.log(sigma_2_current)]),
                                                x_alpha_current]) 
        else:
            if not self.multivariate_x:
                update_block_3 = np.concatenate([np.array([mu_current]),
                                            np.array([gamma_current]),
                                            np.array([np.log(sigma_1_current),
                                                        np.log(sigma_2_current)]),
                                            np.array([np.log(v_current)])])
            else:
                update_block_3 = np.concatenate([np.array([mu_current]),
                                    gamma_current,
                                    np.array([np.log(sigma_1_current),
                                                np.log(sigma_2_current)]),
                                    np.array([np.log(v_current)]),
                                    x_alpha_current])     

        update_block_4 = np.concatenate([update_block_1,
                                         update_block_2,
                                         update_block_3]).copy()
        
        # Block sample counts
        b1_cnts = 0
        b2_cnts = 0
        b3_cnts = 0
        b4_cnts = 0
        print("Track Block: ", self.track_block)
        track_block = self.track_block
        for mc_i in range(self.n_steps): 
            
            block_selector = np.random.choice(len(self.param_sizes_prop), 1, p=self.param_sizes_prop)[0]

            if block_selector == 0:
                
                # Project down x_vals
                if self.multivariate_x:
                    x_proj_current = project_x(self.x_vals, x_alpha_current)
                else:
                    x_proj_current = None

                #### BLOCK 1 Sampler ####
                ###### Sample w1  #######
                
                # First update the block to take in changes from other blocks
                if track_block:
                    update_block_1 = w1_knot_points_current

                am_lamb_block1 = np.exp(log_am_lamb_block1)
                proposal_vec  = np.random.multivariate_normal(update_block_1,  
                                                            am_lamb_block1*am_cov_block1+np.eye(len(mu_block1))*self.eps_1)
                
                w1_knot_prop = proposal_vec
                
                self.prop_check1.append(proposal_vec)
                
                
                
                # Update w1_sample
                w1_approx_prop, lp_w1_prop = calc_mixture_knot_approx_marginalized(w1_knot_prop,
                                                                                    a_kappa=alpha_kappa,
                                                                                    b_kappa=beta_kappa,
                                                                                    tau_grid=tau_grid_expanded,
                                                                                    A_g_matrices=A_matrices_G,
                                                                                    cov_mat_knot_store=cov_matrices_G,
                                                                                    lambda_grid_log_prob=lambda_grid_log_prob)

                # Calc likelihood
                ll_prop = eval_ll(self.y_vals_true,
                                self.x_vals,
                                w_samples_1=w1_approx_prop,
                                w_samples_2=w2_approx_current,
                                sigma_1=sigma_1_current,
                                sigma_2=sigma_2_current,
                                tau_grid=self.tau_grid,
                                tau_grid_expanded=self.tau_grid_expanded,
                                mu=mu_current,
                                gamma=gamma_current,
                                base_quantile_mean=self.base_quantile_mean,
                                base_quantile_sd=self.base_quantile_sd,
                                base_quantile_v=v_current,
                                base_quantile_dist=self.base_quantile_dist,
                                splice=self.splice,
                                multi_var=self.multivariate_x,
                                proj_x=x_proj_current)

                ll_curr = eval_ll(self.y_vals_true,
                                self.x_vals,
                                w_samples_1=w1_approx_current,
                                w_samples_2=w2_approx_current,
                                sigma_1=sigma_1_current,
                                sigma_2=sigma_2_current,
                                tau_grid=self.tau_grid,
                                tau_grid_expanded=self.tau_grid_expanded,
                                mu=mu_current,
                                gamma=gamma_current,
                                base_quantile_mean=self.base_quantile_mean,
                                base_quantile_sd=self.base_quantile_sd,
                                base_quantile_v=v_current,
                                base_quantile_dist=self.base_quantile_dist,
                                splice=self.splice,
                                multi_var=self.multivariate_x,
                                proj_x=x_proj_current)    

                self.ll_check_blk1.append((ll_prop, ll_curr))
                # Take metropolis step
                
                # Prior Prob
                log_prior_prop = lp_w1_prop
                log_prior_current = lp_w1_current
                
                # Proposal Props
                log_proposal_prop = 0
                log_proposal_current = 0
                
                
                trans_weight_1 = (ll_prop + log_prior_prop + log_proposal_current) - \
                                (ll_curr + log_prior_current + log_proposal_prop) 
                
                if np.isnan(trans_weight_1):
                    print('Transition Error Block 1')
                    if mc_i > 5000:
                        trans_weight_1 = -1e99
                    #trans_weight_1 = (log_prior_prop + log_proposal_current) - (log_prior_current + log_proposal_prop)  # Dont rely on likelihood

                a_1 = np.exp(min(0,  trans_weight_1))
                
                self.a_check_1.append(a_1)
                #print(a)

                if np.random.uniform(0,1) < a_1:  # Accepted
                    # Update AM Sampling parameters
                    w1_knot_points_current = w1_knot_prop

                    # Update approximation
                    w1_approx_current = w1_approx_prop
                    lp_w1_current = lp_w1_prop

                    self.block_1_accept_cnts += 1
                else: 
                    w1_knot_points_current = w1_knot_points_current
                    w1_approx_current = w1_approx_current
                    lp_w1_current = lp_w1_current
                    rho_current = rho_current
                
                
                # Update AM sampling parameters
                update_block_1 = w1_knot_points_current
                
                
                # Adaptive metropolis update for block 1
                log_am_lamb_block1 = log_am_lamb_block1 + step_sizes_1[b1_cnts]*(a_1 - self.a_target_1)
                #print(log_am_lamb_block1)
                mu_block1_update = mu_block1 + step_sizes_1[b1_cnts]*(update_block_1 - mu_block1)
                
                am_cov_block1 =  am_cov_block1 + \
                                step_sizes_1[b1_cnts]*( (update_block_1 - mu_block1).reshape(-1,1) @\
                                                            ((update_block_1 - mu_block1).reshape(-1,1).T) - am_cov_block1)
                
                mu_block1 = mu_block1_update
                

                # Update block counts
                b1_cnts += 1
            
            
            elif block_selector == 1:
                #### BLOCK 2 Sampler ####
                ###### Sample w2  #######
                # Project down x_vals
                if self.multivariate_x:
                    x_proj_current = project_x(self.x_vals, x_alpha_current)
                else:
                    x_proj_current = None
                
                # Update block to take in changes from other blocks
                if track_block:
                    update_block_2 = w2_knot_points_current

                am_lamb_block2 = np.exp(log_am_lamb_block2)
                proposal_vec  = np.random.multivariate_normal(update_block_2,  
                                                            am_lamb_block2*am_cov_block2+np.eye(len(mu_block2))*self.eps_2)
                
                w2_knot_prop = proposal_vec
                
                self.prop_check2.append(proposal_vec)
                
                
                # Update w2_sample
                w2_approx_prop, lp_w2_prop = calc_mixture_knot_approx_marginalized(w2_knot_prop,
                                                                                    a_kappa=alpha_kappa,
                                                                                    b_kappa=beta_kappa,
                                                                                    tau_grid=tau_grid_expanded,
                                                                                    A_g_matrices=A_matrices_G,
                                                                                    cov_mat_knot_store=cov_matrices_G,
                                                                                    lambda_grid_log_prob=lambda_grid_log_prob)

                # Calc likelihood
                ll_prop = eval_ll(self.y_vals_true,
                                self.x_vals,
                                w_samples_1=w1_approx_current,
                                w_samples_2=w2_approx_prop,
                                sigma_1=sigma_1_current,
                                sigma_2=sigma_2_current,
                                tau_grid=self.tau_grid,
                                tau_grid_expanded=self.tau_grid_expanded,
                                mu=mu_current,
                                gamma=gamma_current,
                                base_quantile_mean=self.base_quantile_mean,
                                base_quantile_sd=self.base_quantile_sd,
                                base_quantile_v=v_current,
                                base_quantile_dist=self.base_quantile_dist,
                                splice=self.splice,
                                multi_var=self.multivariate_x,
                                proj_x=x_proj_current)

                ll_curr = eval_ll(self.y_vals_true,
                                self.x_vals,
                                w_samples_1=w1_approx_current,
                                w_samples_2=w2_approx_current,
                                sigma_1=sigma_1_current,
                                sigma_2=sigma_2_current,
                                tau_grid=self.tau_grid,
                                tau_grid_expanded=self.tau_grid_expanded,
                                mu=mu_current,
                                gamma=gamma_current,
                                base_quantile_mean=self.base_quantile_mean,
                                base_quantile_sd=self.base_quantile_sd,
                                base_quantile_v=v_current,
                                base_quantile_dist=self.base_quantile_dist,
                                splice=self.splice,
                                multi_var=self.multivariate_x,
                                proj_x=x_proj_current)    

                self.ll_check_blk2.append((ll_prop, ll_curr))
                
                # Take metropolis step
                
                # Prior Prob
                log_prior_prop = lp_w2_prop
                log_prior_current = lp_w2_current
                
                # Proposal Props
                log_proposal_prop = 0
                log_proposal_current = 0
                
                
                trans_weight_2 = (ll_prop + log_prior_prop + log_proposal_current) - \
                                (ll_curr + log_prior_current + log_proposal_prop) 
                
                if np.isnan(trans_weight_2):
                    print('Transition Error Block 2')
                    if mc_i > 5000:  # But after some step we want to ignore these transitions completely
                        trans_weight_2 = -1e99
                    #trans_weight_2 = (log_prior_prop + log_proposal_current) - (log_prior_current + log_proposal_prop)  # Dont rely on likelihood
                a_2 = np.exp(min(0,  trans_weight_2))
                
                self.a_check_2.append(a_2)
                #print(a)

                if np.random.uniform(0,1) < a_2:  # Accepted
                    # Update AM Sampling parameters
                    w2_knot_points_current = w2_knot_prop

                    # Update approximation
                    w2_approx_current = w2_approx_prop
                    lp_w2_current = lp_w2_prop

                    self.block_2_accept_cnts += 1
                else: 
                    w2_knot_points_current = w2_knot_points_current
                    w2_approx_current = w2_approx_current
                    lp_w2_current = lp_w2_current
                    rho_current = rho_current
                

                # Update AM sampling parameters
                update_block_2 = w2_knot_points_current
                
                
                # Adaptive metropolis update for block 1
                log_am_lamb_block2 = log_am_lamb_block2 + step_sizes_2[mc_i]*(a_2 - self.a_target_2)
                #print(log_am_lamb_block1)
                mu_block2_update = mu_block2 + step_sizes_2[mc_i]*(update_block_2 - mu_block2)
                
                am_cov_block2 =  am_cov_block2 + \
                                step_sizes_2[mc_i]*( (update_block_2 - mu_block2).reshape(-1,1) @\
                                                            ((update_block_2 - mu_block2).reshape(-1,1).T) - am_cov_block2)
                
                mu_block2 = mu_block2_update
                
                
                b2_cnts += 1

            elif block_selector == 2:
                ############ BLOCK 3 Sampler #################
                #### Sample mu, gamma, sigma1, sigma2  and DF v if prior used####   
                am_lamb_block3 = np.exp(log_am_lamb_block3)
                #print('Update 3: ', update_block_3.shape)
                #print('Update 3: ', am_cov_block3.shape)

                # Update the block to take in changes from other blocks
                if track_block:
                    if self.prior_on_t:
                        if not self.multivariate_x:
                            update_block_3 = np.concatenate([np.array([mu_current]),
                                                    np.array([gamma_current]),
                                                    np.array([np.log(sigma_1_current),
                                                                np.log(sigma_2_current)]),
                                                    np.array([np.log(v_current)])])

                        else:
                            update_block_3 = np.concatenate([np.array([mu_current]),
                                                    gamma_current,
                                                    np.array([np.log(sigma_1_current),
                                                                np.log(sigma_2_current)]),
                                                    np.array([np.log(v_current)]),
                                                    x_alpha_current])     

                    else:
                        if not self.multivariate_x:
                            update_block_3 = np.concatenate([np.array([mu_current]),
                                                    np.array([gamma_current]),
                                                    np.array([np.log(sigma_1_current),
                                                                np.log(sigma_2_current)])])
                        
                        else:
                            update_block_3 = np.concatenate([np.array([mu_current]),
                                                    gamma_current,
                                                    np.array([np.log(sigma_1_current),
                                                                np.log(sigma_2_current)]),
                                                    x_alpha_current])


                proposal_vec3  = np.random.multivariate_normal(update_block_3,
                                                            am_lamb_block3*am_cov_block3+self.eps_3*np.eye(len(mu_block3)) )
                self.blk3_check.append((mu_block3, am_lamb_block3, am_cov_block3))

                mu_prop = proposal_vec3[0]

                # Extract parameters
                if self.multivariate_x:
                    gamma_prop = proposal_vec3[1:1+self.n_covar]
                
                    sigma_1_prop = np.exp(proposal_vec3[1+self.n_covar])
                    sigma_2_prop = np.exp(proposal_vec3[1+self.n_covar+1])

                    if self.prior_on_t:
                        v_prop = np.exp(proposal_vec3[1+self.n_covar+2])
                        x_alpha_prop = proposal_vec3[1+self.n_covar+3 : 1+self.n_covar+3 + self.n_covar]
                    else:
                        x_alpha_prop = proposal_vec3[1+self.n_covar+2 : 1+self.n_covar+2 + self.n_covar]

                else:
                    gamma_prop = proposal_vec3[1]
                    sigma_1_prop = np.exp(proposal_vec3[2])
                    sigma_2_prop = np.exp(proposal_vec3[3])

                    if self.prior_on_t:
                        v_prop = np.exp(proposal_vec3[4])

                if self.multivariate_x:
                    #print("length prop_vec 3: ", len(proposal_vec3))
                    #print(len(x_alpha_prop))
                    x_proj_prop = project_x(self.x_vals, x_alpha_prop)
                    x_proj_current = project_x(self.x_vals, x_alpha_current)
                else:
                    x_proj_prop = None
                    x_proj_current = None

                
                self.prop_check3.append((proposal_vec3, mu_block3))
                

                # Calc likelihood
                ll_prop = eval_ll(self.y_vals_true,
                                self.x_vals,
                                w_samples_1=w1_approx_current,
                                w_samples_2=w2_approx_current,
                                sigma_1=sigma_1_prop,
                                sigma_2=sigma_2_prop,
                                tau_grid=self.tau_grid,
                                tau_grid_expanded=self.tau_grid_expanded,
                                mu=mu_prop,
                                gamma=gamma_prop,
                                base_quantile_mean=self.base_quantile_mean,
                                base_quantile_sd=self.base_quantile_sd,
                                base_quantile_v=v_prop,
                                base_quantile_dist=self.base_quantile_dist,
                                splice=self.splice,
                                multi_var=self.multivariate_x,
                                proj_x=x_proj_prop)

                ll_curr = eval_ll(self.y_vals_true,
                                self.x_vals,
                                w_samples_1=w1_approx_current,
                                w_samples_2=w2_approx_current,
                                sigma_1=sigma_1_current,
                                sigma_2=sigma_2_current,
                                tau_grid=self.tau_grid,
                                tau_grid_expanded=self.tau_grid_expanded,
                                mu=mu_current,
                                gamma=gamma_current,
                                base_quantile_mean=self.base_quantile_mean,
                                base_quantile_sd=self.base_quantile_sd,
                                base_quantile_v=v_current,
                                base_quantile_dist=self.base_quantile_dist,
                                splice=self.splice,
                                multi_var=self.multivariate_x,
                                proj_x=x_proj_current)    

                self.ll_check_blk3.append((ll_prop, ll_curr))

                # Take metropolis step

                # Prior Probs
                log_prior_prop = student_t.logpdf(mu_prop, df=1,loc=self.mu_mean, scale=self.mu_scale) +\
                                gamma.logpdf(sigma_1_prop**2,  a=self.sigma_sq_1_a, scale=1/self.sigma_sq_1_b) +\
                                gamma.logpdf(sigma_2_prop**2,  a=self.sigma_sq_2_a, scale=1/self.sigma_sq_2_b)
                
                log_prior_current = student_t.logpdf(mu_current, df=1,loc=self.mu_mean, scale=self.mu_scale)+\
                                    gamma.logpdf(sigma_1_current**2,  a=self.sigma_sq_1_a, scale=1/self.sigma_sq_1_b) +\
                                    gamma.logpdf(sigma_2_current**2,  a=self.sigma_sq_2_a, scale=1/self.sigma_sq_2_b)
                
                if self.prior_on_t:
                    # Surya logistic
                    #log_prior_prop += logistic.logpdf(v_prop,loc=0,scale=1/6)
                    #log_prior_current += logistic.logpdf(v_current,loc=0,scale=1/6)
                    
                    # Gamma(2,10) https://statmodeling.stat.columbia.edu/2015/05/17/do-we-have-any-recommendations-for-priors-for-student_ts-degrees-of-freedom-parameter/

                    log_prior_prop += gamma.logpdf(v_prop, a=3, scale=1/2)
                    log_prior_current += gamma.logpdf(v_current,a=3, scale=1/2)

                if self.multivariate_x:
                    log_prior_prop += multivariate_t.logpdf(x_alpha_prop, loc = np.zeros(self.n_covar),
                                             shape=np.eye(self.n_covar),
                                             df=1)
                    log_prior_current += multivariate_t.logpdf(x_alpha_current, loc = np.zeros(self.n_covar),
                                             shape=np.eye(self.n_covar),
                                             df=1)
                    
                    # Log pdf for gamma
                    log_prior_prop += np.sum([student_t.logpdf(gp,df=1,
                                                               loc=self.gamma_mean,
                                                                 scale=self.gamma_scale) for gp in gamma_prop])
                    log_prior_current += np.sum([student_t.logpdf(gp,df=1,
                                                loc=self.gamma_mean,
                                                    scale=self.gamma_scale) for gp in gamma_current])
                else:
                    # Just single logpdf for gamma
                    log_prior_prop += student_t.logpdf(gamma_prop,df=1,loc=self.gamma_mean, scale=self.gamma_scale)
                    log_prior_current += student_t.logpdf(gamma_current, df=1,loc=self.gamma_mean, scale=self.gamma_scale)


                # Adjust for jacobian of transformation
                log_proposal_prop = 0  - sigma_1_prop  - sigma_2_prop
                log_proposal_current = 0 - sigma_1_current  - sigma_2_current

                if self.prior_on_t:
                    log_proposal_prop -= v_prop
                    log_proposal_current -= v_current
                
                
                #if (ll_prop + log_prior_prop + log_proposal_current) > 0:
                #    break
                trans_weight_3 = (ll_prop + log_prior_prop + log_proposal_current) \
                                - (ll_curr +log_prior_current + log_proposal_prop)
                if np.isnan(trans_weight_3):
                    print('Transition Error Block 3')
                    if mc_i > 5000:
                        trans_weight_3 = -1e99
                    #trans_weight_3 = (log_prior_prop + log_proposal_current) - (log_prior_current + log_proposal_prop)  # Dont rely on likelihood

                a_3 = np.exp(min(0,  trans_weight_3 ))
                
                self.a_check_3.append((a_3, 
                                ll_prop + log_prior_prop + log_proposal_current,
                                ll_curr +log_prior_current + log_proposal_prop))
                
                #print(a)
                if np.random.uniform(0,1) < a_3:
                    mu_current = mu_prop
                    gamma_current = gamma_prop
                    sigma_1_current = sigma_1_prop
                    sigma_2_current = sigma_2_prop
                    
                    if self.prior_on_t:
                        v_current = v_prop

                    if self.multivariate_x:
                        x_alpha_current = x_alpha_prop

                    self.block_3_accept_cnts += 1
                else: 
                    mu_current = mu_current
                    gamma_current = gamma_current
                    sigma_1_current = sigma_1_current
                    sigma_2_current = sigma_2_current

                    if self.prior_on_t:
                        v_current = v_current
                    
                    if self.multivariate_x:
                        x_alpha_current = x_alpha_current
                
                
                # Update AM block 3 sampling parameters
                if self.prior_on_t:
                    if not self.multivariate_x:
                        update_block_3 = np.concatenate([np.array([mu_current]),
                                                np.array([gamma_current]),
                                                np.array([np.log(sigma_1_current),
                                                            np.log(sigma_2_current)]),
                                                np.array([np.log(v_current)])])

                    else:
                        update_block_3 = np.concatenate([np.array([mu_current]),
                                                gamma_current,
                                                np.array([np.log(sigma_1_current),
                                                            np.log(sigma_2_current)]),
                                                np.array([np.log(v_current)]),
                                                x_alpha_current])     

                else:
                    if not self.multivariate_x:
                        update_block_3 = np.concatenate([np.array([mu_current]),
                                                np.array([gamma_current]),
                                                np.array([np.log(sigma_1_current),
                                                            np.log(sigma_2_current)])])
                    
                    else:
                        update_block_3 = np.concatenate([np.array([mu_current]),
                                                gamma_current,
                                                np.array([np.log(sigma_1_current),
                                                            np.log(sigma_2_current)]),
                                                x_alpha_current])
                    
                # Adaptive metropolis update for block 1
                log_am_lamb_block3 = log_am_lamb_block3 + step_sizes_3[mc_i]*(a_3 - self.a_target_3)
                #print(log_am_lamb_block1)
                mu_block3_update = mu_block3 + step_sizes_3[mc_i]*(update_block_3 - mu_block3)
                
                am_cov_block3 =  am_cov_block3 + \
                                step_sizes_3[mc_i]*( (update_block_3 - mu_block3).reshape(-1,1) @\
                                                            ((update_block_3 - mu_block3).reshape(-1,1).T) - am_cov_block3)
                
                mu_block3 = mu_block3_update.copy()
                self.cov_store_3.append(am_cov_block3)
                b3_cnts += 1


            else: # Block 4
                ############ BLOCK 4 Sampler #################
                #### Sample w1, w2, mu, gamma, sigma1 and sigma2  ####   
                am_lamb_block4 = np.exp(log_am_lamb_block4)
                #print("Block 4: ", update_block_4.shape)
                #print("Block 4: ", am_cov_block4.shape)

                # Update block to take in changes from other blocks
                if track_block:
                    if not self.prior_on_t:
                        if not self.multivariate_x:
                            update_block_4 = np.concatenate([w1_knot_points_current,
                                                            w2_knot_points_current,
                                                            np.array([mu_current]),
                                                            np.array([gamma_current]),
                                                            np.array([np.log(sigma_1_current),
                                                                        np.log(sigma_2_current)])])
                        else:
                            update_block_4 = np.concatenate([w1_knot_points_current,
                                                            w2_knot_points_current,
                                                            np.array([mu_current]),
                                                            gamma_current,
                                                            np.array([np.log(sigma_1_current),
                                                                        np.log(sigma_2_current)]),
                                                            x_alpha_current])  
                        

                    else:
                        if not self.multivariate_x:
                            update_block_4 = np.concatenate([w1_knot_points_current,
                                                            w2_knot_points_current,
                                                            np.array([mu_current]),
                                                            np.array([gamma_current]),
                                                            np.array([np.log(sigma_1_current),
                                                                        np.log(sigma_2_current)]),
                                                            np.array([np.log(v_current)])])
                        else:
                            update_block_4 = np.concatenate([w1_knot_points_current,
                                                            w2_knot_points_current,
                                                            np.array([mu_current]),
                                                            gamma_current,
                                                            np.array([np.log(sigma_1_current),
                                                                        np.log(sigma_2_current)]),
                                                            np.array([np.log(v_current)]),
                                                            x_alpha_current])


                proposal_vec4  = np.random.multivariate_normal(update_block_4,
                                                            am_lamb_block4*am_cov_block4+self.eps_4*np.eye(len(mu_block4)) )
                self.blk4_check.append((mu_block4, am_lamb_block4, am_cov_block4))

                w1_knot_prop = proposal_vec4[0:self.m]
                w2_knot_prop = proposal_vec4[self.m: self.m+self.m]
                
                # Update w1_sample
                w1_approx_prop, lp_w1_prop = calc_mixture_knot_approx_marginalized(w1_knot_prop,
                                                                                    a_kappa=alpha_kappa,
                                                                                    b_kappa=beta_kappa,
                                                                                    tau_grid=tau_grid_expanded,
                                                                                    A_g_matrices=A_matrices_G,
                                                                                    cov_mat_knot_store=cov_matrices_G,
                                                                                    lambda_grid_log_prob=lambda_grid_log_prob)


                # Update w2_sample
                w2_approx_prop, lp_w2_prop = calc_mixture_knot_approx_marginalized(w2_knot_prop,
                                                                                    a_kappa=alpha_kappa,
                                                                                    b_kappa=beta_kappa,
                                                                                    tau_grid=tau_grid_expanded,
                                                                                    A_g_matrices=A_matrices_G,
                                                                                    cov_mat_knot_store=cov_matrices_G,
                                                                                    lambda_grid_log_prob=lambda_grid_log_prob)


                # Extract parameters
                mu_prop = proposal_vec4[self.m+self.m]
                
                if self.multivariate_x:
                    gamma_prop = proposal_vec4[self.m+self.m+1: self.m+self.m+1+self.n_covar]
                
                    sigma_1_prop = np.exp(proposal_vec4[self.m+self.m+1+self.n_covar])
                    sigma_2_prop = np.exp(proposal_vec4[self.m+self.m+1+self.n_covar+1])

                    if self.prior_on_t:
                        v_prop = np.exp(proposal_vec4[self.m+self.m+1+self.n_covar+2])
                        x_alpha_prop = proposal_vec4[self.m+self.m+1+self.n_covar+3 : self.m+self.m+1+self.n_covar+3 + self.n_covar]
                    else:
                        x_alpha_prop = proposal_vec4[self.m+self.m+1+self.n_covar+2 : self.m+self.m+1+self.n_covar+2 + self.n_covar]

                else:
                    gamma_prop = proposal_vec4[self.m+self.m+1]
                    sigma_1_prop = np.exp(proposal_vec4[self.m+self.m+2])
                    sigma_2_prop = np.exp(proposal_vec4[self.m+self.m+3])

                    if self.prior_on_t:
                        v_prop = np.exp(proposal_vec4[self.m+self.m+4])

                if self.multivariate_x:

                    x_proj_prop = project_x(self.x_vals, x_alpha_prop)
                    x_proj_current = project_x(self.x_vals, x_alpha_current)
                else:
                    x_proj_prop = None
                    x_proj_current = None

                """
                mu_prop = proposal_vec4[self.m+self.m]
                gamma_prop = proposal_vec4[self.m+self.m+1]
                
                sigma_1_prop = np.exp(proposal_vec4[self.m+self.m+2])
                sigma_2_prop = np.exp(proposal_vec4[self.m+self.m+3])
                
                if self.prior_on_t:
                    v_prop = np.exp(proposal_vec4[self.m+self.m+4])
                    if self.multivariate_x:
                        x_alpha_prop = proposal_vec4[self.m+self.m+5:]
                        x_vals_used_prop = project_x(self.x_vals, x_alpha_prop)
                        x_vals_used_current = project_x(self.x_vals, x_alpha_current)
                    else:
                        x_vals_used_prop = self.x_vals
                        x_vals_used_current = self.x_vals
                else:
                    if self.multivariate_x:
                        x_alpha_prop = proposal_vec4[self.m+self.m+4:]  # Position is one before if no prior on t used
                        x_vals_used_prop = project_x(self.x_vals, x_alpha_prop)
                        x_vals_used_current = project_x(self.x_vals, x_alpha_current)
                    else:
                        x_vals_used_prop = self.x_vals
                        x_vals_used_current = self.x_vals
                """

                self.prop_check4.append((proposal_vec4, mu_block4))
                

                # Calc likelihood
                ll_prop = eval_ll(self.y_vals_true,
                                self.x_vals,
                                w_samples_1=w1_approx_prop,
                                w_samples_2=w2_approx_prop,
                                sigma_1=sigma_1_prop,
                                sigma_2=sigma_2_prop,
                                tau_grid=self.tau_grid,
                                tau_grid_expanded=self.tau_grid_expanded,
                                mu=mu_prop,
                                gamma=gamma_prop,
                                base_quantile_mean=self.base_quantile_mean,
                                base_quantile_sd=self.base_quantile_sd,
                                base_quantile_v=v_prop,
                                base_quantile_dist=self.base_quantile_dist,
                                splice=self.splice,
                                multi_var=self.multivariate_x,
                                proj_x=x_proj_prop)

                ll_curr = eval_ll(self.y_vals_true,
                                self.x_vals,
                                w_samples_1=w1_approx_current,
                                w_samples_2=w2_approx_current,
                                sigma_1=sigma_1_current,
                                sigma_2=sigma_2_current,
                                tau_grid=self.tau_grid,
                                tau_grid_expanded=self.tau_grid_expanded,
                                mu=mu_current,
                                gamma=gamma_current,
                                base_quantile_mean=self.base_quantile_mean,
                                base_quantile_sd=self.base_quantile_sd,
                                base_quantile_v=v_current,
                                base_quantile_dist=self.base_quantile_dist,
                                splice=self.splice,
                                multi_var=self.multivariate_x,
                                proj_x=x_proj_current)    

                self.ll_check_blk4.append((ll_prop, ll_curr))

                # Take metropolis step
                            
                # Prior Probs
                # Note about scaling standard logistic dist, v ~ logis(0,1) -> v/6 ~ logis(0,1/6)
                # https://stats.libretexts.org/Bookshelves/Probability_Theory/Probability_Mathematical_Statistics_and_Stochastic_Processes_(Siegrist)/05%3A_Special_Distributions/5.29%3A_The_Logistic_Distribution
                
                log_prior_prop = student_t.logpdf(mu_prop, df=1,loc=self.mu_mean, scale=self.mu_scale) +\
                                gamma.logpdf(sigma_1_prop**2,  a=self.sigma_sq_1_a, scale=1/self.sigma_sq_1_b) +\
                                gamma.logpdf(sigma_2_prop**2,  a=self.sigma_sq_2_a, scale=1/self.sigma_sq_2_b) 
                
                log_prior_current = student_t.logpdf(mu_current, df=1,loc=self.mu_mean, scale=self.mu_scale)+\
                                    gamma.logpdf(sigma_1_current**2,  a=self.sigma_sq_1_a, scale=1/self.sigma_sq_1_b) +\
                                    gamma.logpdf(sigma_2_current**2,  a=self.sigma_sq_2_a, scale=1/self.sigma_sq_2_b) 
                
                # Add in prior for quantile function degress of freedom
                if self.prior_on_t:
                    # Surya logistic
                    #log_prior_prop += logistic.logpdf(v_prop,loc=0,scale=1/6)
                    #log_prior_current += logistic.logpdf(v_current,loc=0,scale=1/6)

                    # Gamma(2,10)
                    log_prior_prop += gamma.logpdf(v_prop, a=3, scale=1/2)
                    log_prior_current += gamma.logpdf(v_current,a=3, scale=1/2)
                
                # Add in prior for multivariate x alpha if needed
                if self.multivariate_x:
                    log_prior_prop += multivariate_t.logpdf(x_alpha_prop, loc = np.zeros(self.n_covar),
                                             shape=np.eye(self.n_covar),
                                             df=1)
                    log_prior_current += multivariate_t.logpdf(x_alpha_current, loc = np.zeros(self.n_covar),
                                             shape=np.eye(self.n_covar),
                                             df=1)

                    # Log pdf for gamma multivariate independent
                    log_prior_prop += np.sum([student_t.logpdf(gp,df=1,
                                                               loc=self.gamma_mean,
                                                                 scale=self.gamma_scale) for gp in gamma_prop])
                    log_prior_current += np.sum([student_t.logpdf(gp,df=1,
                                                loc=self.gamma_mean,
                                                    scale=self.gamma_scale) for gp in gamma_current])
                else:
                    # Just single logpdf for gamma
                    log_prior_prop += student_t.logpdf(gamma_prop,df=1,loc=self.gamma_mean, scale=self.gamma_scale)
                    log_prior_current += student_t.logpdf(gamma_current, df=1,loc=self.gamma_mean, scale=self.gamma_scale)


                # Get proposal probabilities (Adjusting for transformations for sigma1, sigma2 and v (if used))
                log_proposal_prop = 0  - sigma_1_prop  - sigma_2_prop 
                log_proposal_current = 0 - sigma_1_current  - sigma_2_current 
                                                                
                if self.prior_on_t:
                    
                    log_proposal_prop -= v_prop
                    log_proposal_current -= v_current
                
                #if (ll_prop + log_prior_prop + log_proposal_current) > 0:
                #    break
                
                trans_weight_4 = (ll_prop + log_prior_prop + log_proposal_current) -\
                      (ll_curr + log_prior_current + log_proposal_prop)

                if np.isnan(trans_weight_4):
                    print('Transition Error Block 4')
                    if mc_i > 3000:
                        trans_weight_4 = -1e99
                    #trans_weight_4 = (log_prior_prop + log_proposal_current) - (log_prior_current + log_proposal_prop)

                a_4 = np.exp(min(0,  trans_weight_4 ))
                
                self.a_check_4.append([a_4, 
                                ll_prop + log_prior_prop + log_proposal_current,
                                ll_curr + log_prior_current + log_proposal_prop,
                                (ll_prop, log_prior_prop, log_proposal_current),
                                (ll_curr, log_prior_current, log_proposal_prop)])
                
                #print(a)
                if np.random.uniform(0,1) < a_4:
                    # Update w1, w2
                    w1_knot_points_current = w1_knot_prop
                    w2_knot_points_current = w2_knot_prop

                    # Update approximation
                    w1_approx_current = w1_approx_prop
                    w2_approx_current = w2_approx_prop
                    
                    # update log pdfs w1, w2
                    lp_w1_current = lp_w1_prop
                    lp_w2_current = lp_w2_prop

                    # Update other parameters
                    mu_current = mu_prop
                    gamma_current = gamma_prop
                    sigma_1_current = sigma_1_prop
                    sigma_2_current = sigma_2_prop
                    
                    if self.prior_on_t:
                        v_current = v_prop
                    
                    if self.multivariate_x:
                        x_alpha_current = x_alpha_prop

                    self.block_4_accept_cnts += 1
                else: 
                    # Update w1, w2
                    w1_knot_points_current = w1_knot_points_current
                    w2_knot_points_current = w2_knot_points_current

                    # Update approximation
                    w1_approx_current = w1_approx_current
                    w2_approx_current = w2_approx_current
                    
                    # update log pdfs w1, w2
                    lp_w1_current = lp_w1_current
                    lp_w2_current = lp_w2_current

                    # Update other parameters
                    mu_current = mu_current
                    gamma_current = gamma_current
                    sigma_1_current = sigma_1_current
                    sigma_2_current = sigma_2_current

                    if self.prior_on_t:
                        v_current = v_current

                    if self.multivariate_x:
                        x_alpha_current = x_alpha_current
                
                # Update AM block 4 sampling parameters
                if not self.prior_on_t:
                    if not self.multivariate_x:
                        update_block_4 = np.concatenate([w1_knot_points_current,
                                                        w2_knot_points_current,
                                                        np.array([mu_current]),
                                                        np.array([gamma_current]),
                                                        np.array([np.log(sigma_1_current),
                                                                    np.log(sigma_2_current)])])
                    else:
                        update_block_4 = np.concatenate([w1_knot_points_current,
                                                        w2_knot_points_current,
                                                        np.array([mu_current]),
                                                        gamma_current,
                                                        np.array([np.log(sigma_1_current),
                                                                    np.log(sigma_2_current)]),
                                                        x_alpha_current])  
                    

                else:
                    if not self.multivariate_x:
                        update_block_4 = np.concatenate([w1_knot_points_current,
                                                        w2_knot_points_current,
                                                        np.array([mu_current]),
                                                        np.array([gamma_current]),
                                                        np.array([np.log(sigma_1_current),
                                                                    np.log(sigma_2_current)]),
                                                        np.array([np.log(v_current)])])
                    else:
                        update_block_4 = np.concatenate([w1_knot_points_current,
                                                        w2_knot_points_current,
                                                        np.array([mu_current]),
                                                        gamma_current,
                                                        np.array([np.log(sigma_1_current),
                                                                    np.log(sigma_2_current)]),
                                                        np.array([np.log(v_current)]),
                                                        x_alpha_current])

                
                # Adaptive metropolis update for block 1
                log_am_lamb_block4 = log_am_lamb_block4 + step_sizes_4[b4_cnts]*(a_4 - self.a_target_4)
                #print(log_am_lamb_block1)
                mu_block4_update = mu_block4 + step_sizes_4[b4_cnts]*(update_block_4 - mu_block4)
                
                am_cov_block4 =  am_cov_block4 + \
                                step_sizes_4[b4_cnts]*( (update_block_4 - mu_block4).reshape(-1,1) @\
                                                            ((update_block_4 - mu_block4).reshape(-1,1).T) - am_cov_block4)
                
                mu_block4 = mu_block4_update.copy()
                self.cov_store_4.append(am_cov_block4)
                b4_cnts += 1

            # Store log prob
            self.log_score.append(ll_curr)

            #### Store generated variables ####
            # Knots
            self.w1_knot_store.append(w1_knot_points_current)
            self.w2_knot_store.append(w2_knot_points_current)
            # Approximated variables
            self.w1_approx_store.append(w1_approx_current)
            self.w2_approx_store.append(w2_approx_current)

            # Also store w1+w2 as w (for plotting)
            self.w_approx_store.append(np.concatenate([w1_approx_current,
                                                    w2_approx_current]))


            self.sigma_1_store.append(sigma_1_current)
            self.sigma_2_store.append(sigma_2_current)
            self.mu_store.append(mu_current)
            self.gamma_store.append(gamma_current)
            self.x_alpha_store.append(x_alpha_current)

            if self.prior_on_t: 
                self.v_store.append(v_current)

            if ((mc_i%1000 == 0) and (mc_i != 0)) and (self.verbose==True):
                e = time.time()
                sample_prop = np.array([b1_cnts, b2_cnts, b3_cnts, b4_cnts])/mc_i
                print('Step: ', mc_i, ' Time Taken: ', e-s,
                      'Block 1 Accept: ', 100*self.block_1_accept_cnts/b1_cnts,
                      ' Block 2 Accept: ',100*self.block_2_accept_cnts/b2_cnts,
                      ' Block 3 Accept: ', 100*self.block_3_accept_cnts/b3_cnts,
                      ' Block 4 Accept: ', 100*self.block_4_accept_cnts/b4_cnts,
                      ' Sampled Prop: ',sample_prop)
                s = time.time()
            
            elif mc_i%5000 == 0:
                e = time.time()
                print('Step: ', mc_i, ' Time Taken: ', e-s)
                s = time.time()

            """
            if (mc_i%100 == 0) and (self.verbose==True):
                print('Lambda Current: ', lambd_current)
                print('Mu Current: ', mu_current)
                print('Gamma Current: ', gamma_current)
                print('Sigma 1 Current: ', sigma_1_current)
                print('Sigma 2 Current: ', sigma_2_current)
                print('v Current: ', v_current)
            """

        output_dict = {'w1_knot':self.w1_knot_store,
                       'w2_knot':self.w2_knot_store,
                        'w': self.w_approx_store,
                       'lambda': self.lambda_store,
                        'mu': self.mu_store,
                         'gamma': self.gamma_store,
                         'sigma_1': self.sigma_1_store,
                         'sigma_2':self.sigma_2_store,
                         'v': self.v_store,
                         'X_alpha': self.x_alpha_store}
        
        return output_dict
    


#### N - Step Block metropolis Sampler ####
# For us in EPA - QR code #
def block_metropolis_steps_QR(y_vals_true,
                            x_vals,
                            phi_current,
                            cov_mat_knots_init,
                            A_matrices_G,
                            cov_matrices_G,
                            lambda_grid_log_prob,
                            n_steps=200,
                            n_adapt = 100,
                            C_list= [0.3,0.3,0.3,0.3],
                            lambda_step_sizes = [3,3,3,3],
                            alpha_step_sizes = [0.3,0.3,0.3,0.3],
                            a_targets = [0.28,0.28,0.28,0.28],
                            tau_grid_expanded = np.arange(-0.01,1.02,0.01),
                            tau_grid = np.arange(0.01,1.0,0.01),     
                            knot_points_grid = np.arange(0.1,1,0.1),
                            alpha_kappa = 5,
                            beta_kappa = 1/3,
                            base_quantile_mean=0.0,
                            base_quantile_sd=1.0,
                            base_quantile_v=1.0,
                            base_quantile_dist='norm',
                            prior_on_t=False,
                            splice=True):
    
    # Multivariate track
    multivariate_x = False
    if len(x_vals.shape) > 1:
        if x_vals.shape[1] > 1:
            multivariate_x = True

    if multivariate_x:
        n_covar = x_vals.shape[1]

    # Set sampler parameters
    C_1,C_2,C_3,C_4 = C_list
    
    lambda_step_size_1, lambda_step_size_2,\
        lambda_step_size_3,lambda_step_size_4 = lambda_step_sizes
    
    alpha_step_size_1, alpha_step_size_2,\
        alpha_step_size_3,alpha_step_size_4 = alpha_step_sizes
    
    a_target_1, a_target_2, a_target_3, a_target_4 = a_targets
    
    eps_1 = 1e-5
    eps_2 = 1e-5
    eps_3 = 1e-5
    eps_4 = 1e-4
    
    am_lamb_block1_init = 0.1,
    am_lamb_block2_init = 0.5,
    am_lamb_block3_init = 0.5,
    am_lamb_block4_init = 0.5,
    
    
    # Extract Current samples from phi (note phi stores actual parameters)
    # Phi - w1, w2, mu, gamma, sigma_1, sigma_2, v
    #m = (len(phi_current) - 4)// 2
    if multivariate_x:
        if prior_on_t:
            m = (len(phi_current)-4 - 2*n_covar)//2
        else:
            m = (len(phi_current)-3 - 2*n_covar)//2
    else:
        if prior_on_t:
            m = (len(phi_current)-5)//2
        else:
            m = (len(phi_current)-4)//2
    
    w1_knot_points_current = phi_current[0:m]
    w2_knot_points_current = phi_current[m:2*m]
    mu_current = phi_current[2*m]

    if multivariate_x:
        gamma_current = phi_current[2*m+1: 2*m+1+n_covar]
        sigma_1_current = phi_current[2*m+1+n_covar] #np.exp(phi_current[2*m+2]) # TODO: Potential bug no need to exponentiate
        sigma_2_current = phi_current[2*m+1+n_covar + 1] #np.exp(phi_current[2*m+3])
        
        if prior_on_t:
            v_current = phi_current[2*m+1+n_covar + 2]
            x_alpha_current = phi_current[2*m+1+n_covar + 3 : 2*m+1+n_covar + 3 + n_covar]  # Fix

        else:
            v_current = base_quantile_v
            v_prop = base_quantile_v
            x_alpha_current = phi_current[2*m+1+n_covar + 2 : 2*m+1+n_covar + 2 +n_covar]  # Fix
        
        #if np.any(np.isnan(x_alpha_current)):
        #    print("NaN in x_alpha_current detected: ")
        #    print('x_alpha_comp: ', x_alpha_current)


    else:
        gamma_current = phi_current[2*m+1]
        sigma_1_current = phi_current[2*m+2] #np.exp(phi_current[2*m+2]) # TODO: Potential bug no need to exponentiate
        sigma_2_current = phi_current[2*m+3] #np.exp(phi_current[2*m+3])
        
        if prior_on_t:
            v_current = phi_current[2*m+4]
        else:
            v_current = base_quantile_v
            v_prop = base_quantile_v

    # Generate sample of GP approx
    w1_approx_current, lp_w1_current = calc_mixture_knot_approx_marginalized(w1_knot_points_current,
                                                                            a_kappa=alpha_kappa,
                                                                            b_kappa=beta_kappa,
                                                                            tau_grid=tau_grid_expanded,
                                                                            A_g_matrices=A_matrices_G,
                                                                            cov_mat_knot_store=cov_matrices_G,
                                                                            lambda_grid_log_prob=lambda_grid_log_prob)

    w2_approx_current, lp_w2_current = calc_mixture_knot_approx_marginalized(w2_knot_points_current,
                                                                            a_kappa=alpha_kappa,
                                                                            b_kappa=beta_kappa,
                                                                            tau_grid=tau_grid_expanded,
                                                                            A_g_matrices=A_matrices_G,
                                                                            cov_mat_knot_store=cov_matrices_G,
                                                                            lambda_grid_log_prob=lambda_grid_log_prob)


    # Extract etas
    #eta_1_comp = eta1_mat[c_name-1,:]
    #eta_2_comp = eta2_mat[c_name-1,:]


    ### initialise adaptive metropolis (Every gibbs step we re-init)

    # Initial step sizes
    step_sizes_1 = C_1/(np.arange(1,n_steps+10)**alpha_step_size_1)
    step_sizes_2 = C_2/(np.arange(1,n_steps+10)**alpha_step_size_2)
    step_sizes_3 = C_3/(np.arange(1,n_steps+10)**alpha_step_size_3)
    step_sizes_4 = C_4/(np.arange(1,n_steps+10)**alpha_step_size_4)
    
    # Block 1
    am_lamb_block1 = am_lamb_block1_init
    log_am_lamb_block1 = np.log(am_lamb_block1)
    am_cov_block1 = block_diag(cov_mat_knots_init)

    mu_block1 = w1_knot_points_current

    # Block 2
    am_lamb_block2 = am_lamb_block2_init
    log_am_lamb_block2 = np.log(am_lamb_block2)
    am_cov_block2 = block_diag(cov_mat_knots_init)

    mu_block2 = w2_knot_points_current

    # Block 3
    am_lamb_block3 = am_lamb_block3_init
    log_am_lamb_block3 = np.log(am_lamb_block3)
    if not prior_on_t:
        if not multivariate_x:
            am_cov_block3 = np.diag([np.sqrt(5),np.sqrt(5),1,1])
            mu_block3= np.concatenate([np.array([mu_current]),
                                        np.array([gamma_current]),
                                        np.array([np.log(sigma_1_current),
                                                    np.log(sigma_2_current)])])
        
        else:
            am_cov_block3 = np.diag([np.sqrt(5)]+n_covar*[np.sqrt(5)]+[1,1]+n_covar*[1])
            mu_block3= np.concatenate([np.array([mu_current]),
                            gamma_current,
                            np.array([np.log(sigma_1_current),
                                        np.log(sigma_2_current)]),
                            x_alpha_current])


    else:
        if not multivariate_x:
            am_cov_block3 = np.diag([np.sqrt(5),np.sqrt(5),1,1, 1])
            mu_block3= np.concatenate([np.array([mu_current]),
                                        np.array([gamma_current]),
                                        np.array([np.log(sigma_1_current),
                                                    np.log(sigma_2_current)]),
                                        np.array([np.log(v_current)])])

        else:
            am_cov_block3 = np.diag([np.sqrt(5)] + n_covar*[np.sqrt(5)] + [1,1,1]+n_covar*[1])
            mu_block3= np.concatenate([np.array([mu_current]),
                                    gamma_current,
                                    np.array([np.log(sigma_1_current),
                                                np.log(sigma_2_current)]),
                                    np.array([np.log(v_current)]),
                                    x_alpha_current])

    # Block 4
    am_lamb_block4 = am_lamb_block4_init
    log_am_lamb_block4 = np.log(am_lamb_block4)
    am_cov_block4 = block_diag(cov_mat_knots_init,
                               cov_mat_knots_init,am_cov_block3)  # 9 + 9 + 4 dim diagonal covariance matrix
    if not prior_on_t:
        if not multivariate_x:
            mu_block4= np.concatenate([w1_knot_points_current,
                                    w2_knot_points_current,
                                    np.array([mu_current]),
                                        np.array([gamma_current]),
                                        np.array([np.log(sigma_1_current),
                                                    np.log(sigma_2_current)])])
        
        else:
            mu_block4= np.concatenate([w1_knot_points_current,
                        w2_knot_points_current,
                        np.array([mu_current]),
                            gamma_current,
                            np.array([np.log(sigma_1_current),
                                        np.log(sigma_2_current)]),
                        x_alpha_current])

    else:
        if not multivariate_x:
            mu_block4= np.concatenate([w1_knot_points_current,
                            w2_knot_points_current,
                            np.array([mu_current]),
                                np.array([gamma_current]),
                                np.array([np.log(sigma_1_current),
                                            np.log(sigma_2_current)]),
                                np.array([np.log(v_current)])])
        else:
            mu_block4= np.concatenate([w1_knot_points_current,
                            w2_knot_points_current,
                            np.array([mu_current]),
                                gamma_current,
                                np.array([np.log(sigma_1_current),
                                            np.log(sigma_2_current)]),
                                np.array([np.log(v_current)]),
                                x_alpha_current])

    # Set up initial state

    update_block_1 = w1_knot_points_current

    update_block_2 = w2_knot_points_current

    if not prior_on_t:
        if not multivariate_x:
            update_block_3 = np.concatenate([np.array([mu_current]),
                                    np.array([gamma_current]),
                                    np.array([np.log(sigma_1_current),
                                                np.log(sigma_2_current)])])
        
        else:
            update_block_3 = np.concatenate([np.array([mu_current]),
                                    gamma_current,
                                    np.array([np.log(sigma_1_current),
                                                np.log(sigma_2_current)]),
                                            x_alpha_current]) 
    else:
        if not multivariate_x:
            update_block_3 = np.concatenate([np.array([mu_current]),
                                        np.array([gamma_current]),
                                        np.array([np.log(sigma_1_current),
                                                    np.log(sigma_2_current)]),
                                        np.array([np.log(v_current)])])
        else:
            update_block_3 = np.concatenate([np.array([mu_current]),
                                gamma_current,
                                np.array([np.log(sigma_1_current),
                                            np.log(sigma_2_current)]),
                                np.array([np.log(v_current)]),
                                x_alpha_current])     

    update_block_4 = np.concatenate([update_block_1,
                                     update_block_2,
                                     update_block_3]).copy()

    # Block sample counts
    b1_cnts = 0
    b2_cnts = 0
    b3_cnts = 0
    b4_cnts = 0
    
    if multivariate_x:  
        if prior_on_t:
            param_sizes_prop = np.array([9,9,4+n_covar+n_covar,9+9+4+n_covar+n_covar])
            param_sizes_prop = param_sizes_prop/param_sizes_prop.sum()
        else:
            param_sizes_prop = np.array([9,9,3+n_covar+n_covar,9+9+3+n_covar+n_covar])
            param_sizes_prop = param_sizes_prop/param_sizes_prop.sum()
    else:
        if prior_on_t:
            param_sizes_prop = np.array([9,9,5,(9+9+5)])
            param_sizes_prop = param_sizes_prop/param_sizes_prop.sum()
        else:
            param_sizes_prop = np.array([9,9,4,(9+9+4)])
            param_sizes_prop = param_sizes_prop/param_sizes_prop.sum()


    for mc_i in range(n_steps): 

        block_selector = np.random.choice(len(param_sizes_prop), 1, p=param_sizes_prop)[0]

        if block_selector == 0:

            #### BLOCK 1 Sampler ####
            ###### Sample w1  #######

            # Project down x_vals
            if multivariate_x:
                x_proj_current = project_x(x_vals, x_alpha_current)
                #if np.any(np.isnan(x_proj_current)):
                #    print("NaN in x_proj_current detected 1: ")
                #    print('x_alpha_comp: ', x_alpha_current)
                #    print('x_proj_comp: ', x_proj_current)
            else:
                x_proj_current = None

            #### BLOCK 1 Sampler ####
            ###### Sample w1  #######
            
            # First update the block to take in changes from other blocks
            if True:#track_block:
                update_block_1 = w1_knot_points_current

            am_lamb_block1 = np.exp(log_am_lamb_block1)
            proposal_vec  = np.random.multivariate_normal(update_block_1,  
                                                        am_lamb_block1*am_cov_block1+np.eye(len(mu_block1))*eps_1)

            w1_knot_prop = proposal_vec

            #self.prop_check1.append(proposal_vec)



            # Update w1_sample
            w1_approx_prop, lp_w1_prop = calc_mixture_knot_approx_marginalized(w1_knot_prop,
                                                                                a_kappa=alpha_kappa,
                                                                                b_kappa=beta_kappa,
                                                                                tau_grid=tau_grid_expanded,
                                                                                A_g_matrices=A_matrices_G,
                                                                                cov_mat_knot_store=cov_matrices_G,
                                                                                lambda_grid_log_prob=lambda_grid_log_prob)

            # Calc likelihood
            ll_prop = eval_ll(y_vals_true,
                                    x_vals,
                                    w_samples_1=w1_approx_prop,
                                    w_samples_2=w2_approx_current,
                                    sigma_1=sigma_1_current,
                                    sigma_2=sigma_2_current,
                                    tau_grid=tau_grid,
                                    tau_grid_expanded=tau_grid_expanded,
                                    mu=mu_current,
                                    gamma=gamma_current,
                                    base_quantile_mean=base_quantile_mean,
                                    base_quantile_sd=base_quantile_sd,
                                    base_quantile_v=v_current,
                                    base_quantile_dist=base_quantile_dist,
                                    splice=splice,
                                    proj_x=x_proj_current,
                                    multi_var=multivariate_x)

            ll_curr = eval_ll(y_vals_true,
                                    x_vals,
                                    w_samples_1=w1_approx_current,
                                    w_samples_2=w2_approx_current,
                                    sigma_1=sigma_1_current,
                                    sigma_2=sigma_2_current,
                                    tau_grid=tau_grid,
                                    tau_grid_expanded=tau_grid_expanded,
                                    mu=mu_current,
                                    gamma=gamma_current,
                                    base_quantile_mean=base_quantile_mean,
                                    base_quantile_sd=base_quantile_sd,
                                    base_quantile_v=v_current,
                                    base_quantile_dist=base_quantile_dist,
                                    splice=splice,
                                    proj_x=x_proj_current,
                                    multi_var=multivariate_x)    

            #self.ll_check_blk1.append((ll_prop, ll_curr))
            # Take metropolis step

            # Prior Prob
            log_prior_prop = lp_w1_prop
            log_prior_current = lp_w1_current

            # Proposal Props
            log_proposal_prop = 0
            log_proposal_current = 0


            trans_weight_1 = (ll_prop + log_prior_prop + log_proposal_current) - \
                            (ll_curr + log_prior_current + log_proposal_prop) 

            if np.isnan(trans_weight_1): # Occurs when both ll_prop and ll_curr are -np.inf -> Sometimes let these nans go through for better sampling (else forever stuck)
                print('Transition Error Block 1')
                if mc_i > n_adapt:
                    trans_weight_1 = -1e99  # So that it doesnt transition
            a_1 = np.exp(min(0,  trans_weight_1))

            #self.a_check_1.append(a_1)
            #print(a)

            if np.random.uniform(0,1) < a_1:  # Accepted
                # Update AM Sampling parameters
                w1_knot_points_current = w1_knot_prop

                # Update approximation
                w1_approx_current = w1_approx_prop
                lp_w1_current = lp_w1_prop

                #self.block_1_accept_cnts += 1
            else: 
                w1_knot_points_current = w1_knot_points_current
                w1_approx_current = w1_approx_current
                lp_w1_current = lp_w1_current



            # Update AM sampling parameters
            update_block_1 = w1_knot_points_current


            # Adaptive metropolis update for block 1
            log_am_lamb_block1 = log_am_lamb_block1 + step_sizes_1[b1_cnts]*(a_1 - a_target_1)
            #print(log_am_lamb_block1)
            mu_block1_update = mu_block1 + step_sizes_1[b1_cnts]*(update_block_1 - mu_block1)

            am_cov_block1 =  am_cov_block1 + \
                            step_sizes_1[b1_cnts]*( (update_block_1 - mu_block1).reshape(-1,1) @\
                                                        ((update_block_1 - mu_block1).reshape(-1,1).T) - am_cov_block1)

            mu_block1 = mu_block1_update


            # Update block counts
            b1_cnts += 1


        elif block_selector == 1:
            #### BLOCK 2 Sampler ####
            ###### Sample w2  #######
            # Project down x_vals
            if multivariate_x:
                x_proj_current = project_x(x_vals, x_alpha_current)
                #if np.any(np.isnan(x_proj_current)):
                #    print("NaN in x_proj_current detected 2: ")
                #    print('x_alpha_comp: ', x_alpha_current)
                #    print('x_proj_comp: ', x_proj_current)
            else:
                x_proj_current = None
            
            # Update block to take in changes from other blocks
            if True:
                update_block_2 = w2_knot_points_current

            am_lamb_block2 = np.exp(log_am_lamb_block2)
            proposal_vec  = np.random.multivariate_normal(update_block_2,  
                                                        am_lamb_block2*am_cov_block2+np.eye(len(mu_block2))*eps_2)

            w2_knot_prop = proposal_vec

            #self.prop_check2.append(proposal_vec)


            # Update w2_sample
            w2_approx_prop, lp_w2_prop = calc_mixture_knot_approx_marginalized(w2_knot_prop,
                                                                                a_kappa=alpha_kappa,
                                                                                b_kappa=beta_kappa,
                                                                                tau_grid=tau_grid_expanded,
                                                                                A_g_matrices=A_matrices_G,
                                                                                cov_mat_knot_store=cov_matrices_G,
                                                                                lambda_grid_log_prob=lambda_grid_log_prob)

            # Calc likelihood
            ll_prop = eval_ll(y_vals_true,
                                    x_vals,
                                    w_samples_1=w1_approx_current,
                                    w_samples_2=w2_approx_prop,
                                    sigma_1=sigma_1_current,
                                    sigma_2=sigma_2_current,
                                    tau_grid=tau_grid,
                                    tau_grid_expanded=tau_grid_expanded,
                                    mu=mu_current,
                                    gamma=gamma_current,
                                    base_quantile_mean=base_quantile_mean,
                                    base_quantile_sd=base_quantile_sd,
                                    base_quantile_v=v_current,
                                    base_quantile_dist=base_quantile_dist,
                                    splice=splice,
                                    proj_x=x_proj_current,
                                    multi_var=multivariate_x)

            ll_curr = eval_ll(y_vals_true,
                                    x_vals,
                                    w_samples_1=w1_approx_current,
                                    w_samples_2=w2_approx_current,
                                    sigma_1=sigma_1_current,
                                    sigma_2=sigma_2_current,
                                    tau_grid=tau_grid,
                                    tau_grid_expanded=tau_grid_expanded,
                                    mu=mu_current,
                                    gamma=gamma_current,
                                    base_quantile_mean=base_quantile_mean,
                                    base_quantile_sd=base_quantile_sd,
                                    base_quantile_v=v_current,
                                    base_quantile_dist=base_quantile_dist,
                                    splice=splice,
                                    proj_x=x_proj_current,
                                    multi_var=multivariate_x)    

            #self.ll_check_blk2.append((ll_prop, ll_curr))

            # Take metropolis step

            # Prior Prob
            log_prior_prop = lp_w2_prop
            log_prior_current = lp_w2_current

            # Proposal Props
            log_proposal_prop = 0
            log_proposal_current = 0


            trans_weight_2 = (ll_prop + log_prior_prop + log_proposal_current) - \
                            (ll_curr + log_prior_current + log_proposal_prop) 

            if np.isnan(trans_weight_2):
                print('Transition Error Block 2')
                if mc_i > n_adapt:
                    trans_weight_2 = -1e99
            a_2 = np.exp(min(0,  trans_weight_2))

            #self.a_check_2.append(a_2)
            #print(a)

            if np.random.uniform(0,1) < a_2:  # Accepted
                # Update AM Sampling parameters
                w2_knot_points_current = w2_knot_prop

                # Update approximation
                w2_approx_current = w2_approx_prop
                lp_w2_current = lp_w2_prop

                #self.block_2_accept_cnts += 1
            else: 
                w2_knot_points_current = w2_knot_points_current
                w2_approx_current = w2_approx_current
                lp_w2_current = lp_w2_current


            # Update AM sampling parameters
            update_block_2 = w2_knot_points_current


            # Adaptive metropolis update for block 1
            log_am_lamb_block2 = log_am_lamb_block2 + step_sizes_2[mc_i]*(a_2 - a_target_2)
            #print(log_am_lamb_block1)
            mu_block2_update = mu_block2 + step_sizes_2[mc_i]*(update_block_2 - mu_block2)

            am_cov_block2 =  am_cov_block2 + \
                            step_sizes_2[mc_i]*( (update_block_2 - mu_block2).reshape(-1,1) @\
                                                        ((update_block_2 - mu_block2).reshape(-1,1).T) - am_cov_block2)

            mu_block2 = mu_block2_update


            b2_cnts += 1

        elif block_selector == 2:
            ############ BLOCK 3 Sampler #################
            #### Sample mu, gamma, sigma1 and sigma2  ####   
            # take in changes from previous
            if True:
                if prior_on_t:
                    if not multivariate_x:
                        update_block_3 = np.concatenate([np.array([mu_current]),
                                                np.array([gamma_current]),
                                                np.array([np.log(sigma_1_current),
                                                            np.log(sigma_2_current)]),
                                                np.array([np.log(v_current)])])

                    else:
                        update_block_3 = np.concatenate([np.array([mu_current]),
                                                gamma_current,
                                                np.array([np.log(sigma_1_current),
                                                            np.log(sigma_2_current)]),
                                                np.array([np.log(v_current)]),
                                                x_alpha_current])     

                else:
                    if not multivariate_x:
                        update_block_3 = np.concatenate([np.array([mu_current]),
                                                np.array([gamma_current]),
                                                np.array([np.log(sigma_1_current),
                                                            np.log(sigma_2_current)])])
                    
                    else:
                        update_block_3 = np.concatenate([np.array([mu_current]),
                                                gamma_current,
                                                np.array([np.log(sigma_1_current),
                                                            np.log(sigma_2_current)]),
                                                x_alpha_current])


            am_lamb_block3 = np.exp(log_am_lamb_block3)
            proposal_vec3  = np.random.multivariate_normal(update_block_3,
                                                        am_lamb_block3*am_cov_block3+eps_3*np.eye(len(mu_block3)) )
            #self.blk3_check.append((mu_block3, am_lamb_block3, am_cov_block3))

            mu_prop = proposal_vec3[0]

            # Extract parameters
            if multivariate_x:
                gamma_prop = proposal_vec3[1:1+n_covar]
            
                sigma_1_prop = np.exp(proposal_vec3[1+n_covar])
                sigma_2_prop = np.exp(proposal_vec3[1+n_covar+1])

                if prior_on_t:
                    v_prop = np.exp(proposal_vec3[1+n_covar+2])
                    x_alpha_prop = proposal_vec3[1+n_covar+3 : 1+n_covar+3 + n_covar]
                else:
                    x_alpha_prop = proposal_vec3[1+n_covar+2 : 1+n_covar+2 + n_covar]

            else:
                gamma_prop = proposal_vec3[1]
                sigma_1_prop = np.exp(proposal_vec3[2])
                sigma_2_prop = np.exp(proposal_vec3[3])

                if prior_on_t:
                    v_prop = np.exp(proposal_vec3[4])

            if multivariate_x:
                #print("length prop_vec 3: ", len(proposal_vec3))
                #print(len(x_alpha_prop))
                x_proj_prop = project_x(x_vals, x_alpha_prop)
                x_proj_current = project_x(x_vals, x_alpha_current)
                #if np.any(np.isnan(x_proj_current)):
                #    print("NaN in x_proj_current detected 3: ")
                #    print('x_alpha_comp: ', x_alpha_current)
                #    print('x_proj_comp: ', x_proj_current)
                #if np.any(np.isnan(x_proj_prop)):
                #    print("NaN in x_proj_prop detected 3: ")
                #    print('x_alpha_prop: ', x_alpha_prop)
                #    print('x_proj_prop: ', x_proj_prop)
            else:
                x_proj_prop = None
                x_proj_current = None


            #self.prop_check3.append((proposal_vec3, mu_block3))

            # Calc likelihood
            ll_prop = eval_ll(y_vals_true,
                                    x_vals,
                                    w_samples_1=w1_approx_current,
                                    w_samples_2=w2_approx_current,
                                    sigma_1=sigma_1_prop,
                                    sigma_2=sigma_2_prop,
                                    tau_grid=tau_grid,
                                    tau_grid_expanded=tau_grid_expanded,
                                    mu=mu_prop,
                                    gamma=gamma_prop,
                                    base_quantile_mean=base_quantile_mean,
                                    base_quantile_sd=base_quantile_sd,
                                    base_quantile_v=v_prop,
                                    base_quantile_dist=base_quantile_dist,
                                    splice=splice,
                                    proj_x=x_proj_prop,
                                    multi_var=multivariate_x)

            ll_curr = eval_ll(y_vals_true,
                                    x_vals,
                                    w_samples_1=w1_approx_current,
                                    w_samples_2=w2_approx_current,
                                    sigma_1=sigma_1_current,
                                    sigma_2=sigma_2_current,
                                    tau_grid=tau_grid,
                                    tau_grid_expanded=tau_grid_expanded,
                                    mu=mu_current,
                                    gamma=gamma_current,
                                    base_quantile_mean=base_quantile_mean,
                                    base_quantile_sd=base_quantile_sd,
                                    base_quantile_v=v_current,
                                    base_quantile_dist=base_quantile_dist,
                                    splice=splice,
                                    proj_x=x_proj_current,
                                    multi_var=multivariate_x)    

            #self.ll_check_blk3.append((ll_prop, ll_curr))

            # Take metropolis step

            # Prior Probs
            log_prior_prop = student_t.logpdf(mu_prop, df=1,loc=0, scale=1) +\
                            gamma.logpdf(sigma_1_prop**2,  a=2, scale=1/2) +\
                            gamma.logpdf(sigma_2_prop**2,  a=2, scale=1/2) 
                            # student_t.logpdf(gamma_prop,df=1,loc=0, scale=1) +\

                            
            #norm.logpdf(mu_prop,0,10) +\
                            #norm.logpdf(gamma_prop,0,10) +\


            log_prior_current = student_t.logpdf(mu_current, df=1,loc=0, scale=1)+\
                                gamma.logpdf(sigma_1_current**2,  a=2, scale=1/2) +\
                                gamma.logpdf(sigma_2_current**2,  a=2, scale=1/2) 
                              # student_t.logpdf(gamma_current, df=1,loc=0, scale=1)


            if prior_on_t:
                # Surya logistic
                #log_prior_prop += logistic.logpdf(v_prop,loc=0,scale=1/6)
                #log_prior_current += logistic.logpdf(v_current,loc=0,scale=1/6)
                
                # Gamma(2,10) https://statmodeling.stat.columbia.edu/2015/05/17/do-we-have-any-recommendations-for-priors-for-student_ts-degrees-of-freedom-parameter/

                log_prior_prop += gamma.logpdf(v_prop, a=3, scale=1/2)
                log_prior_current += gamma.logpdf(v_current,a=3, scale=1/2)

            if multivariate_x:
                log_prior_prop += multivariate_t.logpdf(x_alpha_prop, loc = np.zeros(n_covar),
                                            shape=np.eye(n_covar),
                                            df=1)
                log_prior_current += multivariate_t.logpdf(x_alpha_current, loc = np.zeros(n_covar),
                                            shape=np.eye(n_covar),
                                            df=1)
                
                # Log pdf for gamma
                log_prior_prop += np.sum([student_t.logpdf(gp,df=1,
                                                            loc=0,
                                                                scale=1) for gp in gamma_prop])
                log_prior_current += np.sum([student_t.logpdf(gp,df=1,
                                            loc=0,
                                                scale=1) for gp in gamma_current])
            else:
                # Just single logpdf for gamma
                log_prior_prop += student_t.logpdf(gamma_prop,df=1,loc=0, scale=1)
                log_prior_current += student_t.logpdf(gamma_current, df=1,loc=0, scale=1)


            # Add transition prob
            log_proposal_prop = 0  - sigma_1_prop  - sigma_2_prop
            log_proposal_current = 0 - sigma_1_current  - sigma_2_current

            if prior_on_t:
                log_proposal_prop -= v_prop
                log_proposal_current -= v_current

            #if (ll_prop + log_prior_prop + log_proposal_current) > 0:
            #    break

            trans_weight_3 = (ll_prop + log_prior_prop + log_proposal_current) \
                            - (ll_curr +log_prior_current + log_proposal_prop)

            if np.isnan(trans_weight_3): # Occurs when both ll_prop and ll_curr are -np.inf
                print('Transition Error Block 3')
                if mc_i > n_adapt:
                    trans_weight_3 = -1e99  # So that it doesnt transition

            a_3 = np.exp(min(0,  trans_weight_3 ))

            #self.a_check_3.append((a_3, 
            #                ll_prop + log_prior_prop + log_proposal_current,
            #                ll_curr +log_prior_current + log_proposal_prop))

            #print(a)
            if np.random.uniform(0,1) < a_3:
                mu_current = mu_prop
                gamma_current = gamma_prop
                sigma_1_current = sigma_1_prop
                sigma_2_current = sigma_2_prop

                if prior_on_t:
                    v_current=v_prop

                if multivariate_x:
                    x_alpha_current = x_alpha_prop
                #self.block_3_accept_cnts += 1
            else: 
                mu_current = mu_current
                gamma_current = gamma_current
                sigma_1_current = sigma_1_current
                sigma_2_current = sigma_2_current

                if prior_on_t:
                    v_current=v_current
                if multivariate_x:
                    x_alpha_current = x_alpha_current

            # Update AM block 3 sampling parameters
            if prior_on_t:
                if not multivariate_x:
                    update_block_3 = np.concatenate([np.array([mu_current]),
                                            np.array([gamma_current]),
                                            np.array([np.log(sigma_1_current),
                                                        np.log(sigma_2_current)]),
                                            np.array([np.log(v_current)])])

                else:
                    update_block_3 = np.concatenate([np.array([mu_current]),
                                            gamma_current,
                                            np.array([np.log(sigma_1_current),
                                                        np.log(sigma_2_current)]),
                                            np.array([np.log(v_current)]),
                                            x_alpha_current])     

            else:
                if not multivariate_x:
                    update_block_3 = np.concatenate([np.array([mu_current]),
                                            np.array([gamma_current]),
                                            np.array([np.log(sigma_1_current),
                                                        np.log(sigma_2_current)])])
                
                else:
                    update_block_3 = np.concatenate([np.array([mu_current]),
                                            gamma_current,
                                            np.array([np.log(sigma_1_current),
                                                        np.log(sigma_2_current)]),
                                            x_alpha_current])


            # Adaptive metropolis update for block 1
            log_am_lamb_block3 = log_am_lamb_block3 + step_sizes_3[mc_i]*(a_3 - a_target_3)
            #print(log_am_lamb_block1)
            mu_block3_update = mu_block3 + step_sizes_3[mc_i]*(update_block_3 - mu_block3)

            am_cov_block3 =  am_cov_block3 + \
                            step_sizes_3[mc_i]*( (update_block_3 - mu_block3).reshape(-1,1) @\
                                                        ((update_block_3 - mu_block3).reshape(-1,1).T) - am_cov_block3)

            mu_block3 = mu_block3_update.copy()
            #self.cov_store_3.append(am_cov_block3)
            b3_cnts += 1


        else: # Block 4
            ############ BLOCK 4 Sampler #################

            ## Update block with other block updates ##
            if True:
                if not prior_on_t:
                    if not multivariate_x:
                        update_block_4 = np.concatenate([w1_knot_points_current,
                                                        w2_knot_points_current,
                                                        np.array([mu_current]),
                                                        np.array([gamma_current]),
                                                        np.array([np.log(sigma_1_current),
                                                                    np.log(sigma_2_current)])])
                    else:
                        update_block_4 = np.concatenate([w1_knot_points_current,
                                                        w2_knot_points_current,
                                                        np.array([mu_current]),
                                                        gamma_current,
                                                        np.array([np.log(sigma_1_current),
                                                                    np.log(sigma_2_current)]),
                                                        x_alpha_current])  
                    

                else:
                    if not multivariate_x:
                        update_block_4 = np.concatenate([w1_knot_points_current,
                                                        w2_knot_points_current,
                                                        np.array([mu_current]),
                                                        np.array([gamma_current]),
                                                        np.array([np.log(sigma_1_current),
                                                                    np.log(sigma_2_current)]),
                                                        np.array([np.log(v_current)])])
                    else:
                        update_block_4 = np.concatenate([w1_knot_points_current,
                                                        w2_knot_points_current,
                                                        np.array([mu_current]),
                                                        gamma_current,
                                                        np.array([np.log(sigma_1_current),
                                                                    np.log(sigma_2_current)]),
                                                        np.array([np.log(v_current)]),
                                                        x_alpha_current])

            #### Sample w1, w2, mu, gamma, sigma1 and sigma2  ####   
            am_lamb_block4 = np.exp(log_am_lamb_block4)
            proposal_vec4  = np.random.multivariate_normal(update_block_4,
                                                        am_lamb_block4*am_cov_block4+eps_4*np.eye(len(mu_block4)) )
            #self.blk4_check.append((mu_block4, am_lamb_block4, am_cov_block4))

            w1_knot_prop = proposal_vec4[0:m]
            w2_knot_prop = proposal_vec4[m: m+m]

            # Update w1_sample
            w1_approx_prop, lp_w1_prop = calc_mixture_knot_approx_marginalized(w1_knot_prop,
                                                                                a_kappa=alpha_kappa,
                                                                                b_kappa=beta_kappa,
                                                                                tau_grid=tau_grid_expanded,
                                                                                A_g_matrices=A_matrices_G,
                                                                                cov_mat_knot_store=cov_matrices_G,
                                                                                lambda_grid_log_prob=lambda_grid_log_prob)


            # Update w2_sample
            w2_approx_prop, lp_w2_prop = calc_mixture_knot_approx_marginalized(w2_knot_prop,
                                                                                a_kappa=alpha_kappa,
                                                                                b_kappa=beta_kappa,
                                                                                tau_grid=tau_grid_expanded,
                                                                                A_g_matrices=A_matrices_G,
                                                                                cov_mat_knot_store=cov_matrices_G,
                                                                                lambda_grid_log_prob=lambda_grid_log_prob)

            mu_prop = proposal_vec4[m+m]
            if multivariate_x:
                gamma_prop = proposal_vec4[m+m+1: m+m+1+n_covar]
            
                sigma_1_prop = np.exp(proposal_vec4[m+m+1+n_covar])
                sigma_2_prop = np.exp(proposal_vec4[m+m+1+n_covar+1])

                if prior_on_t:
                    v_prop = np.exp(proposal_vec4[m+m+1+n_covar+2])
                    x_alpha_prop = proposal_vec4[m+m+1+n_covar+3 : m+m+1+n_covar+3 + n_covar]
                else:                    
                    x_alpha_prop = proposal_vec4[m+m+1+n_covar+2 : m+m+1+n_covar+2 + n_covar]

            else:
                gamma_prop = proposal_vec4[m+m+1]
                sigma_1_prop = np.exp(proposal_vec4[m+m+2])
                sigma_2_prop = np.exp(proposal_vec4[m+m+3])

                if prior_on_t:
                    v_prop = np.exp(proposal_vec4[m+m+4])

            if multivariate_x:

                x_proj_prop = project_x(x_vals, x_alpha_prop)
                x_proj_current = project_x(x_vals, x_alpha_current)
                #if np.any(np.isnan(x_proj_current)):
                #    print("NaN in x_proj_current detected 4: ")
                #    print('x_alpha_comp: ', x_alpha_current)
                #    print('x_proj_comp: ', x_proj_current)
                #if np.any(np.isnan(x_proj_prop)):
                #    print("NaN in x_proj_prop detected 4: ")
                #    print('x_alpha_prop: ', x_alpha_prop)
                #    print('x_proj_prop: ', x_proj_prop)
            else:
                x_proj_prop = None
                x_proj_current = None


            #self.prop_check4.append((proposal_vec4, mu_block4))


            # Calc likelihood
            ll_prop = eval_ll(y_vals_true,
                                    x_vals,
                                    w_samples_1=w1_approx_prop,
                                    w_samples_2=w2_approx_prop,
                                    sigma_1=sigma_1_prop,
                                    sigma_2=sigma_2_prop,
                                    tau_grid=tau_grid,
                                    tau_grid_expanded=tau_grid_expanded,
                                    mu=mu_prop,
                                    gamma=gamma_prop,
                                    base_quantile_mean=base_quantile_mean,
                                    base_quantile_sd=base_quantile_sd,
                                    base_quantile_v=v_prop,
                                    base_quantile_dist=base_quantile_dist,
                                    splice=splice,
                                    proj_x=x_proj_prop,
                                    multi_var=multivariate_x)

            ll_curr = eval_ll(y_vals_true,
                                    x_vals,
                                    w_samples_1=w1_approx_current,
                                    w_samples_2=w2_approx_current,
                                    sigma_1=sigma_1_current,
                                    sigma_2=sigma_2_current,
                                    tau_grid=tau_grid,
                                    tau_grid_expanded=tau_grid_expanded,
                                    mu=mu_current,
                                    gamma=gamma_current,
                                    base_quantile_mean=base_quantile_mean,
                                    base_quantile_sd=base_quantile_sd,
                                    base_quantile_v=v_current,
                                    base_quantile_dist=base_quantile_dist,
                                    splice=splice,
                                    proj_x=x_proj_current,
                                    multi_var=multivariate_x)    

            #self.ll_check_blk4.append((ll_prop, ll_curr))

            # Take metropolis step

            # Prior Probs
            log_prior_prop = lp_w1_prop+ lp_w2_prop+\
                            student_t.logpdf(mu_prop, df=1,loc=0, scale=1) +\
                            gamma.logpdf(sigma_1_prop**2,  a=2, scale=1/2) +\
                            gamma.logpdf(sigma_2_prop**2,  a=2, scale=1/2) 
            #norm.logpdf(mu_prop,0,10) +\
                            #norm.logpdf(gamma_prop,0,10) +\


            log_prior_current = lp_w1_current + lp_w2_current +\
                                student_t.logpdf(mu_current, df=1,loc=0, scale=1)+\
                                gamma.logpdf(sigma_1_current**2,  a=2, scale=1/2) +\
                                gamma.logpdf(sigma_2_current**2,  a=2, scale=1/2) 
            
            if prior_on_t:
                # Surya logistic
                #log_prior_prop += logistic.logpdf(v_prop,loc=0,scale=1/6)
                #log_prior_current += logistic.logpdf(v_current,loc=0,scale=1/6)

                # Gamma(2,10)
                log_prior_prop += gamma.logpdf(v_prop, a=3, scale=1/2)
                log_prior_current += gamma.logpdf(v_current,a=3, scale=1/2)

            if multivariate_x:
                log_prior_prop += multivariate_t.logpdf(x_alpha_prop, loc = np.zeros(n_covar),
                                            shape=np.eye(n_covar),
                                            df=1)
                log_prior_current += multivariate_t.logpdf(x_alpha_current, loc = np.zeros(n_covar),
                                            shape=np.eye(n_covar),
                                            df=1)

                # Log pdf for gamma multivariate independent
                log_prior_prop += np.sum([student_t.logpdf(gp,df=1,
                                                            loc=0,
                                                                scale=1) for gp in gamma_prop])
                log_prior_current += np.sum([student_t.logpdf(gp,df=1,
                                            loc=0,
                                                scale=1) for gp in gamma_current])
            else:
                # Just single logpdf for gamma
                log_prior_prop += student_t.logpdf(gamma_prop,df=1,loc=0, scale=1)
                log_prior_current += student_t.logpdf(gamma_current, df=1,loc=0, scale=1)


            log_proposal_prop = 0  - sigma_1_prop  - sigma_2_prop
            log_proposal_current = 0 - sigma_1_current  - sigma_2_current

            if prior_on_t:
                log_proposal_prop -= v_prop
                log_proposal_current -= v_current
            #if (ll_prop + log_prior_prop + log_proposal_current) > 0:
            #    break

            trans_weight_4 = (ll_prop + log_prior_prop + log_proposal_current) \
                            - (ll_curr + log_prior_current + log_proposal_prop)
            
            if np.isnan(trans_weight_4): # Occurs when both ll_prop and ll_curr are -np.inf
                print('Transition Error Block 4')
                if mc_i > n_adapt:
                    trans_weight_4 = -1e99  # So that it doesnt transition

            a_4 = np.exp(min(0,  trans_weight_4 ))

            #self.a_check_4.append((a_4, 
            #                ll_prop + log_prior_prop + log_proposal_current,
            #                ll_curr +log_prior_current + log_proposal_prop))

            #print(a)
            if np.random.uniform(0,1) < a_4:
                # Update w1, w2
                w1_knot_points_current = w1_knot_prop
                w2_knot_points_current = w2_knot_prop

                # Update approximation
                w1_approx_current = w1_approx_prop
                w2_approx_current = w2_approx_prop

                # update log pdfs w1, w2
                lp_w1_current = lp_w1_prop
                lp_w2_current = lp_w2_prop

                # Update other parameters
                mu_current = mu_prop
                gamma_current = gamma_prop
                sigma_1_current = sigma_1_prop
                sigma_2_current = sigma_2_prop
          
                if prior_on_t:
                    v_current = v_prop
                
                if multivariate_x:
                    x_alpha_current = x_alpha_prop
                #self.block_4_accept_cnts += 1
            else: 
                # Update w1, w2
                w1_knot_points_current = w1_knot_points_current
                w2_knot_points_current = w2_knot_points_current

                # Update approximation
                w1_approx_current = w1_approx_current
                w2_approx_current = w2_approx_current

                # update log pdfs w1, w2
                lp_w1_current = lp_w1_current
                lp_w2_current = lp_w2_current

                # Update other parameters
                mu_current = mu_current
                gamma_current = gamma_current
                sigma_1_current = sigma_1_current
                sigma_2_current = sigma_2_current

                if prior_on_t:
                    v_current=v_current

                if multivariate_x:
                    x_alpha_current = x_alpha_current

            # Update AM block 4 sampling parameters
            if not prior_on_t:
                if not multivariate_x:
                    update_block_4 = np.concatenate([w1_knot_points_current,
                                                    w2_knot_points_current,
                                                    np.array([mu_current]),
                                                    np.array([gamma_current]),
                                                    np.array([np.log(sigma_1_current),
                                                                np.log(sigma_2_current)])])
                else:
                    update_block_4 = np.concatenate([w1_knot_points_current,
                                                    w2_knot_points_current,
                                                    np.array([mu_current]),
                                                    gamma_current,
                                                    np.array([np.log(sigma_1_current),
                                                                np.log(sigma_2_current)]),
                                                    x_alpha_current])  
                

            else:
                if not multivariate_x:
                    update_block_4 = np.concatenate([w1_knot_points_current,
                                                    w2_knot_points_current,
                                                    np.array([mu_current]),
                                                    np.array([gamma_current]),
                                                    np.array([np.log(sigma_1_current),
                                                                np.log(sigma_2_current)]),
                                                    np.array([np.log(v_current)])])
                else:
                    update_block_4 = np.concatenate([w1_knot_points_current,
                                                    w2_knot_points_current,
                                                    np.array([mu_current]),
                                                    gamma_current,
                                                    np.array([np.log(sigma_1_current),
                                                                np.log(sigma_2_current)]),
                                                    np.array([np.log(v_current)]),
                                                    x_alpha_current])


            # Adaptive metropolis update for block 1
            log_am_lamb_block4 = log_am_lamb_block4 + step_sizes_4[b4_cnts]*(a_4 - a_target_4)
            #print(log_am_lamb_block1)
            mu_block4_update = mu_block4 + step_sizes_4[b4_cnts]*(update_block_4 - mu_block4)

            am_cov_block4 =  am_cov_block4 + \
                            step_sizes_4[b4_cnts]*( (update_block_4 - mu_block4).reshape(-1,1) @\
                                                        ((update_block_4 - mu_block4).reshape(-1,1).T) - am_cov_block4)

            mu_block4 = mu_block4_update.copy()
            #self.cov_store_4.append(am_cov_block4)
            b4_cnts += 1
        
            
    # Get phi - Phi stores parameters in the correct domains.
    if multivariate_x:
        if not prior_on_t:
            phi_out = np.concatenate([w1_knot_points_current,
                                    w2_knot_points_current,
                                    np.array([mu_current]),
                                    gamma_current,
                                    np.aray([sigma_1_current,   
                                            sigma_2_current]),
                                    x_alpha_current])
        else:
            phi_out = np.concatenate([w1_knot_points_current,
                                w2_knot_points_current,
                                np.array([mu_current]),
                                            gamma_current,
                                np.array([sigma_1_current,   
                                        sigma_2_current,
                                        v_current]),
                                x_alpha_current])
    else:
        if not prior_on_t:
            phi_out = np.concatenate([w1_knot_points_current,
                                    w2_knot_points_current,
                                    np.array([mu_current,
                                              gamma_current]),
                                    np.array([sigma_1_current,   
                                            sigma_2_current])])
        else:
            phi_out = np.concatenate([w1_knot_points_current,
                                w2_knot_points_current,
                                np.array([mu_current,
                                          gamma_current]),
                                np.array([sigma_1_current,   
                                        sigma_2_current,
                                        v_current])])


    # Calculate implied eta
    eta_1_out = eta_function_i_vector(tau_input=tau_grid,
                       w_vals=w1_approx_prop,
                       tau_grid=tau_grid_expanded,
                       mean=base_quantile_mean,
                       sd=base_quantile_sd,
                       v=v_current,
                       sigma=sigma_1_current,
                       dist=base_quantile_dist)

    eta_2_out = eta_function_i_vector(tau_input=tau_grid,
                       w_vals=w2_approx_prop,
                       tau_grid=tau_grid_expanded,
                       mean=base_quantile_mean,
                       sd=base_quantile_sd,
                       v=v_current,
                       sigma=sigma_2_current,
                       dist=base_quantile_dist)
        
    return phi_out, eta_1_out, eta_2_out

