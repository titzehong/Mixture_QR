from epa import *
from sample_processing import *
from split_merge import split_merge_step
import numpy as np
from typing import List
import concurrent.futures
import time 



def gibbs_sample_regression(Ys:np.ndarray,
                            Xs:np.ndarray,
                            Zs:np.ndarray,
                            sim_mat:np.ndarray,
                            partition_init: np.ndarray,
                            phi_init: np.ndarray,
                            phi_mean_prior: np.ndarray,
                            phi_cov_prior: np.ndarray,
                            sigma2_reg_init: np.array,
                            v_0:float,
                            sigma2_0:float,
                            tau_init: float,
                            labels_used: List[int],
                            alpha_init: float,
                            delta_init: float,
                            n_gibbs: int=5000,
                            n_burn_in: int=2000,
                            k: int=100,
                            a_alpha: float=1,
                            b_alpha: float=10,
                            a_delta: float=1,
                            b_delta: float=1,
                            a_tau: float=4,
                            b_tau: float=2,
                            w: float=0.5,
                            reordering: bool=True,
                            use_split_merge_step: bool=True,
                            sample_tau: bool=True,
                            n_chain: int=2,
                            opti: bool=True):
                            



    gibbs_args = {'Ys':Ys,
                'Xs':Xs,
                'Zs':Zs,
                'sim_mat':sim_mat,
                'partition_init':partition_init,
                'phi_init':phi_init,
                'phi_mean_prior':phi_mean_prior,
                'phi_cov_prior':phi_cov_prior,
                'sigma2_reg_init':sigma2_reg_init,
                'v_0': v_0,
                'sigma2_0': sigma2_0,
                'tau_init':  tau_init,
                'labels_used':labels_used,
                'alpha_init':alpha_init,
                'delta_init':delta_init,
                'n_gibbs':n_gibbs,
                'n_burn_in':n_burn_in,
                'k':k,
                'a_alpha':a_alpha,
                'b_alpha':b_alpha,
                'a_delta':a_delta,
                'b_delta':b_delta,
                'a_tau': a_tau,
                'b_tau': b_tau,
                'w':0.5,
                'reordering':reordering,
                'use_split_merge_step':use_split_merge_step,
                'sample_tau':sample_tau}


    if n_chain == 1:

        if opti:
            outputs = gibbs_sample_regression_single_thread_opt_parallel(**gibbs_args)
        else:
            print('Only Opti Avail')

    else:

        if opti:
            outputs = []
            
            with concurrent.futures.ProcessPoolExecutor() as executor:

                results = [executor.submit(gibbs_sample_regression_single_thread_opt_parallel, **gibbs_args) for _ in range(n_chain)]
                
                for f in concurrent.futures.as_completed(results):
                    outputs.append(f.result())
        
        else:
            outputs = []
            
            print(print('Only Opti Avail'))
    
    return outputs



def gibbs_sample_regression_single_thread_opt_parallel(Ys:np.ndarray,
                            Xs:np.ndarray,
                            Zs:np.ndarray,
                            sim_mat:np.ndarray,
                            partition_init: np.ndarray,
                            phi_init: np.ndarray,
                            phi_mean_prior: np.ndarray,
                            phi_cov_prior: np.ndarray,
                            sigma2_reg_init: np.array,
                            v_0:float,
                            sigma2_0:float,
                            tau_init: float,
                            labels_used: List[int],
                            alpha_init: float,
                            delta_init: float,
                            n_gibbs: int=5000,
                            n_burn_in: int=2000,
                            k: int=100,
                            a_alpha: float=1,
                            b_alpha: float=10,
                            a_delta: float=1,
                            b_delta: float=1,
                            a_tau: float=4,
                            b_tau: float=2,
                            w: float=0.5,
                            reordering: bool=True,
                            use_split_merge_step: bool=True,
                            sample_tau: bool=True):
                         
    n = len(Ys)
    # intialize 
    names_used = labels_used
    alpha_samp = alpha_init

    delta_samp = delta_init
    order_samp = np.arange(n)
    np.random.shuffle(order_samp)
    phi_samp = phi_init
    partition_samp = partition_init
    tau_samp = tau_init


    #### gibbs sampling hyper parameters
    n_gibbs = n_gibbs
    k = k # no. of numbers to permute order

    # GRW sampler param
    rw_sd = 0.2

    # alpha prior
    a_alpha = a_alpha
    b_alpha = b_alpha
    alpha_bounds = [0,1e99]

    # delta prior
    a_delta = a_delta
    b_delta = b_delta
    
    assert (w<=1) and (w>=0)
    w = w
    delta_bounds = [0,1]

    tau_bounds = [0,1e99]

    # phi / regression prior
    phi_mean_prior = phi_mean_prior
    phi_cov_prior = phi_cov_prior

    sigma2_reg_samp = sigma2_reg_init


    partition_save = []
    alpha_save = []
    delta_save = []
    phi_save = []
    log_prob_save = []
    sigma_reg_save = []
    tau_save = []

    # Track acceptance rates
    split_merge_accept_save = []
    tau_accept_save = []
    delta_accept_save = []
    order_accept_save = []
    alpha_accept_save = []

    
    s1 = time.time()
    # Gibbs loop
    for g in range(n_gibbs+n_burn_in):
        if g%100 == 0:
            e1 = time.time()
            print("Gibbs: ", g, " Time Taken: ", (e1-s1)/60)
            print("Active No Clusters: ", len(np.unique(partition_samp)))
            s1 = time.time()
            
        # Get last term in order samp
        #last_term_samp = order_samp[-1]
        last_term_id = order_samp[-1]#np.where(order_samp == np.max(order_samp))[0][0]
        # Draw sample for the final term
        partition_samp, phi_samp, names_used, \
            partition_factors, sigma2_reg_samp = sample_conditional_i_clust_gibbs_opti_parallel(
                                        i=last_term_id,
                                        order_place=0,
                                        partition=partition_samp,
                                        return_fac=True,
                                        pre_compute_factors=np.zeros(1),
                                        alpha=alpha_samp,
                                        delta=delta_samp,
                                        sim_mat=sim_mat,
                                        order=order_samp,
                                        phi=phi_samp,
                                        y=Ys,
                                        x=Xs,
                                        sigma_reg=sigma2_reg_samp,
                                        v_0=v_0,
                                        sigma2_0=sigma2_0,
                                        names_used=names_used,
                                        phi_base_mean=phi_mean_prior,
                                        phi_base_cov=phi_cov_prior,
                                        reordering=reordering)



        for o in list(range(len(order_samp)-2,-1,-1)):

            term_id = order_samp[o]#np.where(order_samp == o)[0][0]

            partition_samp, phi_samp, names_used,\
                  sigma2_reg_samp= sample_conditional_i_clust_gibbs_opti_parallel(
                                            i=term_id,
                                            order_place=o,
                                            partition=partition_samp,
                                            return_fac=False,
                                            pre_compute_factors=partition_factors,
                                            alpha=alpha_samp,
                                            delta=delta_samp,
                                            sim_mat=sim_mat,
                                            order=order_samp,
                                            phi=phi_samp,
                                            y=Ys,
                                            x=Xs,
                                            sigma_reg=sigma2_reg_samp,
                                            v_0=v_0,
                                            sigma2_0=sigma2_0,
                                            names_used=names_used,
                                            phi_base_mean=phi_mean_prior,
                                            phi_base_cov=phi_cov_prior,
                                            reordering=reordering)

        
        if use_split_merge_step:
        #### Split Merge step ####
            partition_samp,phi_samp, names_used,\
            sigma2_reg_samp, accept_samp = split_merge_step(partition_samp=partition_samp,
                                                        alpha=alpha_samp,
                                                        delta=delta_samp,
                                                        sim_mat=sim_mat,
                                                        order=order_samp,
                                                        phi_samp=phi_samp,
                                                        y=Ys,
                                                        x=Xs, 
                                                        sigma2_reg_samp=sigma2_reg_samp,
                                                        v_0=v_0,
                                                        sigma2_0=sigma2_0,
                                                        names_used=names_used, 
                                                        phi_base_mean=phi_mean_prior,
                                                        phi_base_cov=phi_cov_prior,
                                                        no_intermediate_steps=5)

            if g>n_burn_in:

                split_merge_accept_save.append(accept_samp)


        partition_samp = partition_samp.astype('int')
        # Update phis
        phi_samp = sample_phi(phi_samp, Ys, Xs, partition_samp,
                phi_mean_prior,
                phi_cov_prior,
                sigma2_reg_samp)

        # update sigma
        sigma2_reg_samp = sample_sigma_reg(sigma2_reg_samp,
                            Ys,
                            Xs,
                            partition_samp,
                            v_0,
                            sigma2_0,
                            phi_samp)

        # Sample ordering 
        #order_samp = permute_k(order_samp, k)
        order_samp,order_accept_s = metropolis_step_order(order_current=order_samp,
                                        alpha=alpha_samp,
                                        delta=delta_samp,
                                        partition=partition_samp,
                                        sim_mat=sim_mat,
                                        k=k)
        
        if g>n_burn_in:
            order_accept_save.append(order_accept_s)
        
        #### Sample parameters, alpha, sigma

        alpha_samp, alpha_accept_s = metropolis_step_alpha(alpha_samp,
                                                        rw_sd,
                                                        a_alpha, b_alpha,
                                                        partition_samp,
                                                            delta_samp,
                                                            sim_mat,
                                                            order_samp,
                                                        bounds=alpha_bounds)
        
        if g>n_burn_in:

            alpha_accept_save.append(alpha_accept_s)

        delta_samp, delta_accept_s = metropolis_step_delta(delta_samp, rw_sd,
                                                        a_delta, b_delta, w,
                                                        partition_samp,
                                                            alpha_samp,
                                                            sim_mat,
                                                            order_samp,bounds=delta_bounds)
        
        if g>n_burn_in:

            delta_accept_save.append(delta_accept_s)
        
        # Update Tau + Sim mat
        if sample_tau:
            tau_samp, sim_mat, tau_accept_s = metropolis_step_tau(tau_samp, rw_sd, a_tau, b_tau,
                                partition_samp,
                                    alpha_samp,
                                    delta_samp,
                                    sim_mat,
                                    X=Zs.reshape(-1,1),
                            order=order_samp,bounds=tau_bounds)
            
            if g>n_burn_in:
                tau_accept_save.append(tau_accept_s)
            
        """
        # Calc log prob of result
        log_prob_samp = calc_log_joint(partition=partition_samp,
                                    phi=phi_samp,
                                    y=Ys,
                                    x=Xs,
                                    sim_mat=sim_mat,
                                    order=order_samp,
                                    alpha=alpha_samp,
                                    delta=delta_samp,
                                    sigma_reg = sigma_reg)
        """

        if g>n_burn_in:

            # Save sampled values
            #log_prob_save.append(log_prob_samp)
            partition_save.append(partition_samp)
            alpha_save.append(alpha_samp)
            delta_save.append(delta_samp)
            phi_save.append(phi_samp)
            sigma_reg_save.append(sigma2_reg_samp)
            tau_save.append(tau_samp)


    acceptance_rates = {'Split Merge': np.mean(split_merge_accept_save),
                        'Tau': np.mean(tau_accept_save),
                        'Delta': np.mean(delta_accept_save),
                        'Order': np.mean(order_accept_save),
                        'Alpha': np.mean(alpha_accept_save)}

    acceptance_rates = pd.DataFrame.from_dict(acceptance_rates, orient='index')
    acceptance_rates.columns=['Acceptance Rate']
    acceptance_rates

    return log_prob_save, partition_save, alpha_save, delta_save, tau_save, phi_save, sigma_reg_save, acceptance_rates
        
