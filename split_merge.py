from epa import *
from scipy.stats import gamma, multivariate_normal
import numpy as np

def split_merge_step(partition_samp:np.array,
                    alpha:float,
                    delta:float,
                    sim_mat:np.array,
                    order:np.array,
                    phi_samp:np.array,
                    y:np.array,
                    x:np.array, 
                    sigma2_reg_samp:np.array,
                    v_0:float,
                    sigma2_0:float,
                    names_used:np.array, 
                    phi_base_mean:float,
                    phi_base_cov:np.array,
                    reordering:bool=False,
                    no_intermediate_steps:int=5) -> Tuple[np.array, np.array, List[int]]:
    

    # Choose two candidates and get cluster indices
    i_cand, j_cand = np.random.choice(order, size=2, replace=False)

    c_i = partition_samp[i_cand]
    c_j = partition_samp[j_cand]

    #### Calculate split and merge launch states ####

    names_used = np.unique(partition_samp)

    #### Calculating split launch state #####
    partition_samp_launch_split = partition_samp.copy()
    phi_samp_launch_split = phi_samp.copy()
    sigma2_reg_samp_split = sigma2_reg_samp.copy()

    if c_i == c_j: # Get new cluster name
        new_clust_id = max(names_used) + 1
    else:
        new_clust_id = c_i  # Else use original c_i name

    candidate_choice_split = np.array([new_clust_id, c_j])

    #### Randomly allocate points before restricted gibbs sampling steps ####
    # Extract all points that are co clustered
    neigh_points_split = np.where((partition_samp==c_j) | (partition_samp==new_clust_id))[0]

    # Re allocate points 
    for k in neigh_points_split:

        if k == i_cand:
            partition_samp_launch_split[k] = new_clust_id
        elif k == j_cand:
            partition_samp_launch_split[k] = c_j
        else:
            # Randomly allocate
            partition_samp_launch_split[k] = np.random.choice([new_clust_id, c_j])

    # Resample phi + variance
    phi_new_clust = np.random.multivariate_normal(phi_base_mean,phi_base_cov)
    phi_new_j = np.random.multivariate_normal(phi_base_mean,phi_base_cov)

    gam_alpha_prior = v_0/2
    gam_beta_prior = (v_0*sigma2_0) /2
    sigma_reg_new_clust = gamma.rvs(gam_alpha_prior, scale=1/gam_beta_prior)
    sigma_reg_new_j = gamma.rvs(gam_alpha_prior, scale=1/gam_beta_prior)

    phi_samp_launch_split[c_j-1] =  phi_new_j
    phi_samp_launch_split = np.concatenate([phi_samp_launch_split,np.array([phi_new_clust])])

    sigma2_reg_samp_split[c_j-1] = sigma_reg_new_j
    sigma2_reg_samp_split = np.concatenate([sigma2_reg_samp_split, np.array([sigma_reg_new_clust])])


    #### conduct t restricted gibbs sampling steps ####
    for t in range(no_intermediate_steps):
        for k in neigh_points_split:
            if k in [i_cand, j_cand]:
                continue
            else:
                partition_samp_launch_split, phi_samp_launch_split, names_used, sigma2_reg_samp_split,_ = sample_conditional_i_clust_restricted(i=k,
                                                    partition=partition_samp_launch_split,
                                                    alpha=alpha,
                                                    delta=delta,
                                                    sim_mat=sim_mat,
                                                    order=order,
                                                    phi=phi_samp_launch_split,
                                                    y=y,
                                                    x=x,
                                                    sigma_reg=sigma2_reg_samp_split,
                                                    candidate_choice=candidate_choice_split)

        # Update phis
        phi_samp_launch_split, _ = sample_phi_restricted(phi_samp_launch_split, y, x, partition_samp_launch_split,
                                phi_base_mean,
                                phi_base_cov,
                                sigma2_reg_samp_split,
                                candidate_choice=candidate_choice_split)

        # update sigma
        sigma2_reg_samp_split, _ = sample_sigma_reg_restricted(sigma2_reg_samp_split,
                            y,
                            x,
                            partition_samp_launch_split,
                            v_0,
                            sigma2_0,
                            phi_samp_launch_split,
                            candidate_choice=candidate_choice_split)


    #### Calculating merge launch state #####

    partition_samp_launch_merge = partition_samp.copy()
    phi_samp_launch_merge = phi_samp.copy()
    sigma2_reg_samp_merge = sigma2_reg_samp.copy()


    # Get all neighbouring points
    neigh_points_launch = np.where((partition_samp==c_i) | (partition_samp==c_j))[0]

    new_clust_id = c_j
    candidate_choice_merge = np.array([c_j])

    # Re allocate points 
    for k in neigh_points_launch:
        partition_samp_launch_merge[k] = new_clust_id

    # r restricted gibbs sampling steps to update phi / sigma
    for r in range(no_intermediate_steps):
        
        if r < no_intermediate_steps-1:
            # Update phis
            phi_samp_launch_merge, _ = sample_phi_restricted(phi_samp_launch_merge,
                                                            y, x,
                                                            partition_samp_launch_merge,
                                                            phi_base_mean,
                                                            phi_base_cov,
                                                            sigma2_reg_samp_merge,
                                                            candidate_choice=candidate_choice_merge)

            # update sigma
            sigma2_reg_samp_merge, _ = sample_sigma_reg_restricted(sigma2_reg_samp_merge,
                                y,
                                x,
                                partition_samp_launch_merge,
                                v_0,
                                sigma2_0,
                                phi_samp,
                                candidate_choice=candidate_choice_merge)
        else:
            
            # Take step but save prob of transition
            phi_samp_launch_merge, phi_merge_prob = sample_phi_restricted(phi_samp_launch_merge,
                                                        y, x,
                                                        partition_samp_launch_merge,
                                                        phi_base_mean,
                                                        phi_base_cov,
                                                        sigma2_reg_samp_merge,
                                                        candidate_choice=candidate_choice_merge)

            # update sigma
            sigma2_reg_samp_merge, sigma2_merge_prob = sample_sigma_reg_restricted(sigma2_reg_samp_merge,
                                y,
                                x,
                                partition_samp_launch_merge,
                                v_0,
                                sigma2_0,
                                phi_samp_launch_merge,
                                candidate_choice=candidate_choice_merge)


    #### Final Gibbs step and proposal ####

    if c_i == c_j:  # Propose split
        
        partition_samp_proposal = partition_samp_launch_split.copy()
        phi_samp_proposal = phi_samp_launch_split.copy()
        sigma2_reg_samp_proposal = sigma2_reg_samp_split.copy()

        transition_log_probs = []
        #print('dd2', len(partition_samp_launch_merge))

        for k in neigh_points_split:
            if k in [i_cand, j_cand]:
                continue
            else:
                partition_samp_proposal, phi_samp_proposal, \
                names_used, sigma2_reg_samp_proposal, c_lp  = sample_conditional_i_clust_restricted(i=k,
                                                    partition=partition_samp_proposal,
                                                    alpha=alpha,
                                                    delta=delta,
                                                    sim_mat=sim_mat,
                                                    order=order,
                                                    phi=phi_samp_proposal,
                                                    y=y,
                                                    x=x,
                                                    sigma_reg=sigma2_reg_samp_proposal,
                                                    candidate_choice=candidate_choice_split)
                
                transition_log_probs.append(c_lp)
                
                
        # Update phis
        phi_samp_proposal, phi_lp = sample_phi_restricted(phi_samp_proposal, y, x, partition_samp_proposal,
                                phi_base_mean,
                                phi_base_cov,
                                sigma2_reg_samp_proposal,
                                candidate_choice=candidate_choice_split)

        # update sigma
        sigma2_reg_samp_proposal, sig_lp = sample_sigma_reg_restricted(sigma2_reg_samp_proposal,
                            y,
                            x,
                            partition_samp_proposal,
                            v_0,
                            sigma2_0,
                            phi_samp_proposal,
                            candidate_choice=candidate_choice_split)
        
        transition_log_probs.append(phi_lp)
        transition_log_probs.append(sig_lp)
        
        # Calculate transition probabilities
        # q(y_split| y)
        q_ysplit_y = np.sum(transition_log_probs)
        
        # q(y | y_split)
        q_y_ysplit = phi_merge_prob+sigma2_merge_prob
        
        # P(y_split)
        p_y_split_prior = calc_prior_dist(partition=partition_samp_proposal,
                                        order=order,
                                        sim_mat=sim_mat,
                                        alpha=alpha,
                                        delta=delta,
                                        phi=phi_samp_proposal,
                                        phi_base_mean=phi_base_mean,
                                        phi_base_cov=phi_base_cov,
                                        sigma_reg=sigma2_reg_samp_proposal,
                                        v_0=v_0,
                                        sigma2_0=sigma2_0)
            
        # p(y)
        p_y_prior = calc_prior_dist(partition=partition_samp,
                                    order=order,
                                    sim_mat=sim_mat,
                                    alpha=alpha,
                                    delta=delta,
                                    phi=phi_samp,
                                    phi_base_mean=phi_base_mean,
                                    phi_base_cov=phi_base_cov,
                                    sigma_reg=sigma2_reg_samp,
                                    v_0=v_0,
                                    sigma2_0=sigma2_0)
        
        # L(y_split | data)
        lik_y_split = calc_log_likelihood_all(y=y,
                                            x=x,
                                            phi=phi_samp_proposal,
                                            partition=partition_samp_proposal,
                                            sigma_reg=sigma2_reg_samp_proposal,
                                            model='Gaussian')
        
        
        # L(y | data)
        lik_y = calc_log_likelihood_all(y=y,
                                        x=x,
                                        phi=phi_samp,
                                        partition=partition_samp,
                                        sigma_reg=sigma2_reg_samp,
                                        model='Gaussian')
        
        
        
        # Calculate transition prob
        log_trans_prob = min(np.log(1), (q_y_ysplit + p_y_split_prior + lik_y_split)-(q_ysplit_y+p_y_prior+lik_y))
        trans_prob = np.exp(log_trans_prob)
        
    elif c_i != c_j: # Merge
        
        partition_samp_proposal = partition_samp_launch_merge.copy()
        phi_samp_proposal = phi_samp_launch_merge.copy()
        sigma2_reg_samp_proposal = sigma2_reg_samp_merge.copy()
        
        transition_log_probs = []    
        
        # Take one more gibbs sampling step for phi and sigma
        phi_samp_proposal, phi_merge_prob = sample_phi_restricted(phi_samp_proposal,
                                                y, x,
                                                partition_samp_proposal,
                                                phi_base_mean,
                                                phi_base_cov,
                                                sigma2_reg_samp_proposal,
                                                candidate_choice=candidate_choice_merge)

        # update sigma
        sigma2_reg_samp_merge, sigma2_merge_prob = sample_sigma_reg_restricted(phi_samp_proposal,
                            y,
                            x,
                            partition_samp_proposal,
                            v_0,
                            sigma2_0,
                            phi_samp_proposal,
                            candidate_choice=candidate_choice_merge)
            
        
        
        # Calculate transition probabilities
        # q(y_merge| y)
        q_ymerge_y = phi_merge_prob + sigma2_merge_prob
        
        # q(y | y_merge)
        q_y_ymerge = []
        
        for k in neigh_points_split:
            if k in [i_cand, j_cand]:
                continue
            else:

                chosen_clust = partition_samp[k]

                c_lp  = calc_hypothetical_transition_i(i=k,
                                                    partition=partition_samp_launch_split,
                                                    alpha=alpha,
                                                    delta=delta,
                                                    sim_mat=sim_mat,
                                                    order=order,
                                                    phi=phi_samp_launch_split,
                                                    y=y,
                                                    x=x,
                                                    sigma_reg=sigma2_reg_samp_split,
                                                    candidate_choice=candidate_choice_split,
                                                    chosen_clust=partition_samp[k])

                q_y_ymerge.append(c_lp)    

        q_y_ymerge = np.sum(q_y_ymerge)
        
    
        
        # P(y_merge)
        p_y_merge_prior = calc_prior_dist(partition=partition_samp_proposal,
                                        order=order,
                                        sim_mat=sim_mat,
                                        alpha=alpha,
                                        delta=delta,
                                        phi=phi_samp_proposal,
                                        phi_base_mean=phi_base_mean,
                                        phi_base_cov=phi_base_cov,
                                        sigma_reg=sigma2_reg_samp_proposal,
                                        v_0=v_0,
                                        sigma2_0=sigma2_0)
            
        # p(y)
        p_y_prior = calc_prior_dist(partition=partition_samp,
                                    order=order,
                                    sim_mat=sim_mat,
                                    alpha=alpha,
                                    delta=delta,
                                    phi=phi_samp,
                                    phi_base_mean=phi_base_mean,
                                    phi_base_cov=phi_base_cov,
                                    sigma_reg=sigma2_reg_samp,
                                    v_0=v_0,
                                    sigma2_0=sigma2_0)
        
        # L(y_split | data)
        lik_y_merge = calc_log_likelihood_all(y=y,
                                            x=x,
                                            phi=phi_samp_proposal,
                                            partition=partition_samp_proposal,
                                            sigma_reg=sigma2_reg_samp_proposal,
                                            model='Gaussian')
        
        
        # L(y | data)
        lik_y = calc_log_likelihood_all(y=y,
                                        x=x,
                                        phi=phi_samp,
                                        partition=partition_samp,
                                        sigma_reg=sigma2_reg_samp,
                                        model='Gaussian')
        
        
        
        
        # Calculate transition prob
        log_trans_prob = min(np.log(1), (q_y_ymerge + p_y_merge_prior + lik_y_merge)-(q_ymerge_y+p_y_prior+lik_y))
        trans_prob = np.exp(log_trans_prob)


    # Take metropolis step
    if trans_prob > np.random.uniform(0,1): # Transition

        return partition_samp_proposal, phi_samp_proposal, np.unique(partition_samp_proposal).astype('int'), sigma2_reg_samp_proposal, 1
    
    else:

        return partition_samp, phi_samp, np.unique(partition_samp).astype('int'), sigma2_reg_samp,0



def calc_hypothetical_transition_i(i:int,
                                  partition:np.array,
                                  alpha:float,
                                  delta:float,
                                  sim_mat:np.array,
                                  order:np.array,
                                  phi:np.array,
                                  y:np.array,
                                  x:np.array, 
                                  sigma_reg:np.array,
                                  candidate_choice:np.array,
                                  chosen_clust:int) -> Tuple[np.array, np.array, List[int]]:    
    """
    partition: np.array (1xn) dim array of cluster indices for each data points
    i: which data point to generate candidates for (note only pass in those in S)
    sim_mat: n x n matrix of similarities 
    phis: matrix / vector each column is beta_j
    order: array of 0:(n-1) of any order indicating the (randomly sampled order)
    alpha: alpha parameter of distribution
    delta: delta parameter of distribution
    y: y_data
    x: x_data
    names_used: str of cluster ids used thus far
    reordering: whether to re-order cluster labels to prevent cluster labels from getting large
    """
    
    y_i = y[i]
    x_i = x[i]

    # generate candidate partitions
    candidate_partitions, restricted_clust_ids = generate_candidate_partitions_restricted(i,
                                                                              partition,
                                                                              candidate_choice)
    
    np.where(restricted_clust_ids==chosen_clust)[0][0]
    
    updated_names = candidate_choice.copy()
    # calc log likelihood of each partition
    log_liks = []
    
    for c_name,cp in zip(restricted_clust_ids, candidate_partitions):
        
        # get loglikelihood of the partition
        partition_comp =  partition_log_pdf_fast(cp,
                                        sim_mat, 
                                        order, 
                                        alpha, delta) 


        ll_comp = unit_log_likelihood_pdf(y_i,
                            x_i, phi[c_name-1],  # 1 based indexing for cluster but 0 for phi
                            model='Gaussian',
                            sigma_reg=sigma_reg[c_name-1])
        
        
        # calculate probability of candidate partition and collect
        log_prob_cp = partition_comp + ll_comp

        log_liks.append(log_prob_cp)


    # Collect probabilities and normalize with log-sum-exp trick
    log_liks = np.array(log_liks)
    log_liks = log_liks - logsumexp(log_liks)
    cand_probs = np.exp(log_liks - logsumexp(log_liks))
    
    #print(log_liks)
    
    return log_liks[np.where(restricted_clust_ids==chosen_clust)[0][0]]



def calc_prior_dist(partition,
                    order,
                    sim_mat,
                    alpha,
                    delta,
                    phi,
                    phi_base_mean,
                    phi_base_cov,
                    sigma_reg,
                    v_0,
                    sigma2_0):
    
    prior_log_prob = 0
    
    # Partition
    partition_comp_log_prob = partition_log_pdf_partial_parallel(start_id=0,
                                                                  return_fac=False,
                                                                  partition=partition,
                                                                  sim_mat=sim_mat,
                                                                  order=order,
                                                                  alpha=alpha,
                                                                  delta=delta)
    
    prior_log_prob += partition_comp_log_prob
    
    # Parameters
    cluster_ids = np.unique(partition).astype('int')

    gam_alpha_prior = v_0/2
    gam_beta_prior = (v_0*sigma2_0) /2
    
    prior_params_lp = 0
    for c_id in cluster_ids:
        p_y_split_phi_comp = multivariate_normal.logpdf(phi[c_id-1],
                                               phi_base_mean,
                                                phi_base_cov)
        
        p_y_split_sigma2_comp = gamma.logpdf(sigma_reg[c_id-1],
                                             gam_alpha_prior,
                                             scale=1/gam_beta_prior)
        prior_log_prob += p_y_split_phi_comp
        prior_log_prob += p_y_split_sigma2_comp
        
    return prior_log_prob

def calc_log_likelihood_all(y:float,
                            x:np.array,
                            phi:np.array,
                            partition:np.array,
                            sigma_reg:np.array,
                            model:str='Gaussian'):
    
    total_logpdf = 0
    partition_int = partition.astype('int')
    cluster_ids = np.unique(partition_int)
    
    for c_id in cluster_ids:
        
        cluster_points = np.where(partition_int==c_id)[0]
        phi_c = phi[c_id-1]
        prec_2_c = sigma_reg[c_id-1]

        sigma2_c = (1/prec_2_c)
        
        
        Ys_c = y[cluster_points]
        Xs_c = x[cluster_points]
        
        
        if model == 'Gaussian':
            total_logpdf += multivariate_normal.logpdf(Ys_c,mean=Xs_c@phi_c,
                                                        cov=sigma2_c*np.eye(len(Ys_c)))
    
    return total_logpdf


def sample_sigma_reg_restricted(sigma_reg_cur:np.array,
                     y:np.array,
                     x:np.array,
                     partition:np.array,
                     v_0:float,
                     sigma2_0:float,
                     phi:np.array,
                     candidate_choice:np.array)->np.array:
    """ Samples phi from the full conditional, note this is for a linear regression problem

    Args:
        phi_cur (np.array): current value for phi
        y (float): n dim array of y values 
        x (np.array): n x p array of x values (note add 1s for intercept)
        partition (np.array): current partition
        phi_mean_prior (np.array): prior mean for coefficients
        phi_cov_prior (np.array): prior covariance for coefficients
        sigma_reg (float): error in linear regression outcome
        candidate_choice (np.array): Which cluster indices to update

    Returns:
        np.array: Sample from full conditional of phi
    """
    
    sigma_reg_sample = sigma_reg_cur.copy()


    # Update those in phi cur that are active clusters
    for c_id in candidate_choice:

        y_vals = y[np.where(partition==c_id)[0]]#.reshape(-1,1)
        x_vals = x[np.where(partition==c_id)[0]]
        
        # calc quantities
        err_reg = (y_vals - x_vals @ phi[c_id-1])
        ssr_B = err_reg.T @ err_reg
        n = len(y_vals)
        
        # calc quantities
        gam_alpha_posterior = (v_0+n)/2
        gam_beta_posterior = (v_0*sigma2_0 + ssr_B)/2
        
        # Sample posterior
        clust_sigma_sample = gamma.rvs(a=gam_alpha_posterior, scale=1/gam_beta_posterior)
        #print('sigma_samp: ', clust_sigma_sample)
        #print('g_alpha: ', gam_alpha_posterior)
        #print('g_beta_post: ',gam_beta_posterior)
        # Calc log prob of transition
        sample_lp = gamma.logpdf(clust_sigma_sample, a=gam_alpha_posterior, scale=1/gam_beta_posterior)
        
        sigma_reg_sample[c_id-1] = clust_sigma_sample
        
    return sigma_reg_sample, sample_lp


def sample_phi_restricted(phi_cur:np.array,
               y:np.array,
               x:np.array,
               partition:np.array,
               phi_mean_prior:np.array,
               phi_cov_prior:np.array,
               sigma_reg:np.array,
               candidate_choice:np.array)->np.array:
    """ Samples phi from the full conditional, note this is for a linear regression problem

    Args:
        phi_cur (np.array): current value for phi
        y (float): n dim array of y values 
        x (np.array): n x p array of x values (note add 1s for intercept)
        partition (np.array): current partition
        phi_mean_prior (np.array): prior mean for coefficients
        phi_cov_prior (np.array): prior covariance for coefficients
        sigma_reg (np.array): cluster-wise error in linear regression outcome

    Returns:
        np.array: Sample from full conditional of phi
    """

    phi_sample = phi_cur.copy()
    
    # Update those in phi cur that are active clusters
    for c_id in candidate_choice:
        
        sigma_sq_reg = (1/sigma_reg[c_id-1])

        y_vals = y[np.where(partition==c_id)[0]]#.reshape(-1,1)
        x_vals = x[np.where(partition==c_id)[0]]

        # calc quantities
        xtx = x_vals.T @ x_vals
        xty = x_vals.T @ y_vals

        phi_reg_post_cov_inv = (1/sigma_sq_reg)*xtx  + np.linalg.inv(phi_cov_prior)
        phi_reg_post_cov = np.linalg.inv(phi_reg_post_cov_inv)

        phi_reg_post_mean = phi_reg_post_cov @ (((1/sigma_sq_reg)*xty)  +\
                                                np.linalg.inv(phi_cov_prior) @ phi_mean_prior)

        # Sample posterior
        clust_phi_sample = np.random.multivariate_normal(phi_reg_post_mean.reshape(-1),
                                                        phi_reg_post_cov)
        
        sample_lp = multivariate_normal.logpdf(clust_phi_sample,
                                               mean=phi_reg_post_mean.reshape(-1),
                                               cov=phi_reg_post_cov)

        
        phi_sample[c_id-1] = clust_phi_sample
        
    return phi_sample, sample_lp


def generate_candidate_partitions_restricted(i:int,
                                      partition:np.array,
                                      candidate_choice)->Tuple[List[np.array], List[int], int]:
    """ Used for split merge sampler, generates restricted set of candidate partitions based on cluster indices
    in candidate choice

    Args:
        i (int): which data point (in partition) to generate candidates for
        partition (np.array):  1 x n dim array of cluster indices for each data points
        names_used (List[int]): list of ids which have been used as cluster names so far

    Returns:
        Tuple(List[np.array], List[int], int): candidate_partitions: list of arrays where each array corresponds to candidate partition
                                               clust_ids: the unique cluster labels from the input partition
                                               new_clust_id: the unique cluster labels from the output partition
    """
    
    candidate_partitions = np.zeros((len(candidate_choice), len(partition)))

    # Loop over each unique index and generate a partition where the i-th entry's cluster id is replaced 
    for j in range(len(candidate_choice)):
        cand_part = partition.copy()
        cand_part[i] = candidate_choice[j]

        candidate_partitions[j,:] = cand_part

    return candidate_partitions, candidate_choice

def sample_conditional_i_clust_restricted(i:int,
                                          partition:np.array,
                                          alpha:float,
                                          delta:float,
                                          sim_mat:np.array,
                                          order:np.array,
                                          phi:np.array,
                                          y:np.array,
                                          x:np.array, 
                                          sigma_reg:np.array,
                                          candidate_choice:np.array) -> Tuple[np.array, np.array, List[int]]:    
    """
    partition: np.array (1xn) dim array of cluster indices for each data points
    i: which data point to generate candidates for (note only pass in those in S)
    sim_mat: n x n matrix of similarities 
    phis: matrix / vector each column is beta_j
    order: array of 0:(n-1) of any order indicating the (randomly sampled order)
    alpha: alpha parameter of distribution
    delta: delta parameter of distribution
    y: y_data
    x: x_data
    names_used: str of cluster ids used thus far
    reordering: whether to re-order cluster labels to prevent cluster labels from getting large
    """
    
    y_i = y[i]
    x_i = x[i]

    # generate candidate partitions
    candidate_partitions, restricted_clust_ids = generate_candidate_partitions_restricted(i,
                                                                              partition,
                                                                              candidate_choice)
    
    updated_names = candidate_choice.copy()
    # calc log likelihood of each partition
    log_liks = []
    
    for c_name,cp in zip(restricted_clust_ids, candidate_partitions):
        
        # get loglikelihood of the partition
        partition_comp =  partition_log_pdf_fast(cp,
                                        sim_mat, 
                                        order, 
                                        alpha, delta) 


        ll_comp = unit_log_likelihood_pdf(y_i,
                            x_i, phi[c_name-1],  # 1 based indexing for cluster but 0 for phi
                            model='Gaussian',
                            sigma_reg=sigma_reg[c_name-1])
        
        
        # calculate probability of candidate partition and collect
        log_prob_cp = partition_comp + ll_comp

        log_liks.append(log_prob_cp)


    # Collect probabilities and normalize with log-sum-exp trick
    log_liks = np.array(log_liks)
    cand_probs = np.exp(log_liks - logsumexp(log_liks))

    # sample outcome partition for candidate partitions
    cand_part_choice = np.random.choice(np.arange(len(cand_probs)), p=cand_probs)
    output_partition = candidate_partitions[cand_part_choice]
    #print(restricted_clust_ids)
    #print('Chose: ', restricted_clust_ids[cand_part_choice])

    
    return output_partition, phi, updated_names, sigma_reg, np.log(cand_probs[cand_part_choice])
