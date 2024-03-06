from scipy.stats import norm,multivariate_normal, gamma, beta,multivariate_t
import numpy as np
from sklearn.metrics import pairwise_distances
from typing import List, Tuple
import numba 
import pandas as pd
from numba import prange
import multiprocessing


from Single_Var_QR import block_metropolis_steps_QR
from Single_Var_QR_utils import *

@numba.njit 
def logsumexp(x:np.array)->np.array:
    """ Utility function for log - sum exp trick in normalization of log prob vector

    Args:
        x (np.array): Vector of log probabilities to normalize

    Returns:
        np.array: 
    """
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

@numba.njit 
def exp_sim_func(x1:float, x2:float, tau:float=1) -> float:
    """ Exponential similarity function

    Args:
        x1 (float): first input
        x2 (float): second input

    Returns:
        float: exponential similarity between x1 and x2
    """    
    return np.exp(-tau*np.linalg.norm(x1-x2)**2)



@numba.njit
def partition_log_pdf_fast(partition: np.ndarray=np.array([]),
                      sim_mat: np.ndarray=np.array([[]]),
                      order: np.ndarray=np.array([]),
                      alpha: float=1,
                      delta: float=0) -> float:
    """ Calculates the log probability of a partition given.

    Args:
        partition (np.ndarray): Cluster indices of each point
        sim_mat (np.ndarray): n x n matrix of similarities
        order (np.ndarray): n dim vector of order indicating the randomly sampled order
        alpha (float): alpha parameter
        delta (float): delta parameter

    Returns:
        float: log probability of partition
    """    

    assert len(order) == len(partition), "Length of order and partition inputs not the same"

    q = 1 # No of clusters in t-1 (? one subset in empty set?)
    log_p = 0

    # Loop over each element (based on order) (think this is necessary)
    for t_1, o in enumerate(order):

        t = t_1 + 1 # to match t in formula

        c_o = partition[o]  # cluster membership of current point
        
        if t_1 == 0:
            points_seen = order[0:t_1]
        else:
            points_seen = order[0:t_1-1]

        clust_labels_seen = partition[points_seen]
        q = len(np.unique(clust_labels_seen))
        
        if c_o in clust_labels_seen:  # c_o belongs to existing cluster

            # Extract members of cluster c_o is in 
            cluster_member_position = np.where(clust_labels_seen==c_o)[0]
            cluster_members = order[cluster_member_position]

            # calculate p_t
            point_pairwise_dist = sim_mat[o,:]
            p_t = ((t-1 - delta * q)/(alpha + t-1)) * (np.sum(point_pairwise_dist[cluster_members]) / \
                                                       np.sum(point_pairwise_dist[points_seen]))


        else:  # c_o belongs to new cluster
            # calculate p_t
            p_t = (alpha + delta*q)/(alpha+t-1)


        log_p += np.log(p_t)

    return log_p



def unit_log_likelihood_pdf(y:float,
                            x:np.array,
                            phi:np.array,
                            model:str='Gaussian',
                            sigma_reg:float=1) -> float:
    """_summary_

    Args:
        y (float): y_i value
        x (np.array): X_i value
        phi (np.array): phi_i the parameters of the cluster point i is assigned to
        model (str, optional): Gaussian for gaussian linear model. Defaults to 'Gaussian'.
        sigma_reg (float, optional): sigma of regression likelihood. Defaults to 1.

    Returns:
        float: unit log likelihood
    """

    if model == 'Gaussian':
        model_sigma = np.sqrt(1/sigma_reg)
        return norm.logpdf(y, loc=x@phi, scale=model_sigma)
    else:
        return 1



def remap_partition(partition, relab_dict):
    # from https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
    # Basically maps partition to a new set of indexes based on relab_dict.
    u,inv = np.unique(partition,return_inverse = True)
    recoded_partition = np.array([relab_dict[x] for x in u])[inv].reshape(partition.shape)
    return recoded_partition

def gen_reindex(partition):
    # Get the unique labels of the current partition
    unique_labels = pd.Series(partition).value_counts()
    unique_labels = np.array(unique_labels.index)
    #unique_labels = np.unique(partition)

    # Generate new labels from 1 - max label
    new_labels = np.arange(len(unique_labels)) + 1
    
    return new_labels.astype('int'), unique_labels.astype('int'), dict(zip(unique_labels, new_labels))


def permute_k(current_ordering:np.array ,k:int) -> np.array:
    """ Permutes the first k elements of current_ordering, used to generate prosals for sigma (random ordering)

    Args:
        current_ordering (np.array): Current ordering of variables, array of unique integers with values in 0-(n-1)
        k (int): number of k to permute

    Returns:
        np.array: _description_
    """    
    # permutes first k elements in current_ordering
    # k: int < len(current_ordering)
    # current_ordering np.array of integers signifying order
    
    first_k = current_ordering[0:k]
    remaining_k = current_ordering[k::]

    first_k_permuted = first_k.copy()
    np.random.shuffle(first_k_permuted)

    permuted = np.concatenate([first_k_permuted, remaining_k])
    
    return permuted


def grw_proposal(cur_val: float, sd: float) -> float:
    """ proposal for a gaussian random walk

    Args:
        cur_val (float): current value of the random walk
        sd (float): standard deviation on innovations

    Returns:
        float: a proposed value
    """

    # gaussian random walk proposal distribution
    # proposes normal centered at cur_val with std sd
    return np.random.normal(cur_val, sd)



def delta_log_prior(delta: float, a_delta:float , b_delta:float , w:float)->float:
    """ Calculates the pdf of a delta prior. The prior is a mixture of a beta(a_delta, b_delta) amd a point mass at 0
    with weights (1-w) and w respectivetly.

    Args:
        delta (float): delta value to calculate pdf for
        a_delta (float): a parameter in beta component
        b_delta (float): b parameter in beta component
        w (float): mixture weight

    Returns:
        float: pdf value evaluted at delta
    """

    # mixture prior for beta parameter
    # mixture of a point mass at 0 and a beta(a_delta, b_delta) distribution with mixing weights w
    
    # point mass pdf
    if delta == 0:
        return np.log(w)
    else:
        return np.log(1-w) + beta.logpdf(delta, a_delta, b_delta)
    
    
def metropolis_step_alpha(alpha_cur, sd, a_alpha, b_alpha,
                         partition,
                             delta,
                             sim_mat,
                             order,bounds=None):
    
    # metropolis step for alpha
    
    # Sample proposal
    alpha_prop = grw_proposal(alpha_cur, sd)
    
    if bounds:
        if (alpha_prop<bounds[0]) or (alpha_prop>bounds[1]):
            return alpha_cur, 0
    
    # Get prior log probs (gamma distribution)
    log_prior_cur = gamma.logpdf(alpha_cur, a=a_alpha, scale=1/b_alpha)
    log_prior_prop = gamma.logpdf(alpha_prop, a=a_alpha, scale=1/b_alpha)
    
    # Likelihood values
    log_ll_cur = partition_log_pdf_fast(partition, sim_mat, order, alpha_cur, delta)
    log_ll_prop = partition_log_pdf_fast(partition, sim_mat, order, alpha_prop, delta)
    
    log_num = log_prior_prop + log_ll_prop
    log_denom = log_prior_cur + log_ll_cur
    # get ratio
    mh_ratio = np.exp(log_num - log_denom)
    
    a = min(1, mh_ratio)
    #print(a)
    
    # Accept proposal
    if np.random.uniform(0,1) < a:
        return alpha_prop, 1
    else: # Reject proposa
        return alpha_cur, 0
    

def metropolis_step_delta(delta_cur, sd, a_delta, b_delta, w,
                         partition,
                             alpha,
                             sim_mat,
                             order,bounds=None):
    
    # Sample proposal
    delta_prop = grw_proposal(delta_cur, sd)
    
    if bounds:
        if (delta_prop<bounds[0]) or (delta_prop>bounds[1]):
            return delta_cur, 0
    
    # Get prior probs
    log_prior_cur = delta_log_prior(delta_cur, a_delta=a_delta, b_delta=b_delta, w=w)
    log_prior_prop = delta_log_prior(delta_prop, a_delta=a_delta, b_delta=b_delta, w=w)
    
    # Likelihood values
    log_ll_cur = partition_log_pdf_fast(partition, sim_mat, order, alpha, delta_cur)
    log_ll_prop = partition_log_pdf_fast(partition, sim_mat, order, alpha, delta_prop)

    # get ratio
    log_mh_ratio = (log_prior_prop + log_ll_prop) - (log_prior_cur*log_ll_cur)
    mh_ratio = np.exp(log_mh_ratio)

    a = min(1, mh_ratio)
    #print(a)
    
    # Accept proposal
    if np.random.uniform(0,1) < a:
        return delta_prop, 1
    else: # Reject proposa
        return delta_cur, 0

def metropolis_step_order(order_current:np.ndarray,
                          alpha:float,
                          delta:float,
                          partition:np.ndarray,
                          sim_mat:np.ndarray, k:int) -> np.ndarray:
    """ Metropolis step for sampling the order

    Args:
        order_current (np.ndarray): Current order
        alpha (float): alpha parameter
        delta (float): delta parameter
        partition (np.ndarray): current paratition
        sim_mat (np.ndarray): similarity matrix
        k (int): hyperparameter k, decides how many elements to permute

    Returns:
        np.ndarray: Sampled order
    """    
    
    # calculate log partition prob of curent point
    log_partition_prob_current = partition_log_pdf_fast(partition,
                                                  sim_mat,
                                                  order_current,
                                                  alpha,
                                                  delta) 
    # Sample an order
    order_sample = permute_k(order_current, k)
    
    # calculate log partition prob of proposed point
    log_partition_prob_proposed = partition_log_pdf_fast(partition,
                                                  sim_mat,
                                                  order_sample,
                                                  alpha,
                                                  delta) 
    
    mh_ratio = np.exp(log_partition_prob_proposed - log_partition_prob_current)
    
    # Compare
    a = min(1, mh_ratio)
    
    # Accept proposal
    if np.random.uniform(0,1) < a:
        return order_sample, 1
    else: # Reject proposal
        return order_current, 0



def metropolis_step_tau(tau_cur, sd, a_tau, b_tau,
                         partition,
                         alpha,
                             delta,
                             sim_mat_cur,
                             X,
                             order,bounds=None,):
    
    # metropolis step for alpha
    
    # Sample proposal
    tau_prop = grw_proposal(tau_cur, sd)


    if bounds:
        if (tau_prop<bounds[0]) or (tau_prop>bounds[1]):
            return tau_cur, sim_mat_cur, 0
    

    sim_mat_prop = pairwise_distances(X=X,
                                    metric=exp_sim_func, tau=tau_prop)
    
    # Get prior log probs (gamma distribution)
    log_prior_cur = gamma.logpdf(tau_cur, a=a_tau, scale=1/b_tau)
    log_prior_prop = gamma.logpdf(tau_prop, a=a_tau, scale=1/b_tau)
    
    # Likelihood values
    log_ll_cur = partition_log_pdf_fast(partition, sim_mat_cur, order, alpha, delta)
    log_ll_prop = partition_log_pdf_fast(partition, sim_mat_prop, order, alpha, delta)
    
    log_num = log_prior_prop + log_ll_prop
    log_denom = log_prior_cur + log_ll_cur
    # get ratio
    mh_ratio = np.exp(log_num - log_denom)
    
    a = min(1, mh_ratio)
    #print(a)
    
    # Accept proposal
    if np.random.uniform(0,1) < a:
        return tau_prop, sim_mat_prop, 1
    else: # Reject proposa
        return tau_cur, sim_mat_cur, 0



def sample_phi(phi_cur:np.array,
               y:np.array,
               x:np.array,
               partition:np.array,
               phi_mean_prior:np.array,
               phi_cov_prior:np.array,
               sigma_reg:np.array)->np.array:
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

    active_clust_ids = np.unique(partition)

    phi_sample = phi_cur.copy()
    
    # Update those in phi cur that are active clusters
    for c_id in active_clust_ids:
        
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

        phi_sample[c_id-1] = clust_phi_sample
        
    return phi_sample


def sample_phi_QR(phi_cur:np.array,
               eta1_cur:np.array,
               eta2_cur:np.array,
               y:np.array,
               x:np.array,
              partition:np.array,
              tau_grid:np.array,
              tau_grid_expanded:np.array,
            A_matrices_G:np.array,
            cov_matrices_G:np.array,
            lambda_grid_log_prob:np.array,
            cov_mat_knots_init:np.array,
            n_steps:int=200,
               n_adapt:int=100,
               base_quantile_mean:float=0.0,
               base_quantile_sd:float=1.0,
               base_quantile_v:float=1.0,
               base_quantile_dist:str='norm',
               prior_on_t:bool=False,
               longi_x:bool=False)->np.array:
    """ Samples phi for simultaneous quantile regression model using block metropolis QR sampler

    Args:
        phi_cur (np.array): current value for phi
        y (float): n dim array of y values 
        x (np.array): n vector of values (univariate) /  n x p array of x values (multivar)  / n x T x p array of values (longitudinal multivar)
        partition (np.array): current partition
        phi_mean_prior (np.array): prior mean for coefficients
        phi_cov_prior (np.array): prior covariance for coefficients
        sigma_reg (np.array): cluster-wise error in linear regression outcome
        longi_x (bool): Whether x is longitudinal

    Returns:
        np.array: Sample from full conditional of phi
    """

    active_clust_ids = np.unique(partition)
    phi_sample = phi_cur.copy()
    eta1_sample = eta1_cur.copy()
    eta2_sample = eta2_cur.copy()
    
    # Update those in phi cur that are active clusters
    for c_id in active_clust_ids:

        #print("Updating: ", c_id)
        
        y_vals = y[np.where(partition==c_id)[0]]#.reshape(-1,1)
        x_vals = x[np.where(partition==c_id)[0]]

        if longi_x:
            n_covar = x_vals.shape[2]
            # Reshape x_vals from (N_partition, T, p) -> (N_parition*T , p)
            x_vals = x_vals.reshape(-1, n_covar)
            # reshape y_vals from (N_partition , T) -> N_partition * T
            y_vals = y_vals.reshape(-1)
        
        # Get paramaters
        # Extract Current samples
        phi_clust = phi_sample[c_id-1,:] # Extract c_id-1 row of phi matrix
        
        
        # Update using block metropolis
        phi_update, eta_1_update, eta_2_update = block_metropolis_steps_QR(y_vals,
                                                    x_vals,
                                                    phi_clust,
                                                    n_steps=n_steps,
                                                    n_adapt = n_adapt,
                                                    C_list= [0.1,0.1,0.1,0.1],
                                                    lambda_step_sizes = [3,3,3,3],
                                                    alpha_step_sizes = [0.3,0.3,0.3,0.3],
                                                    a_targets = [0.28,0.28,0.28,0.28],
                                                    tau_grid_expanded = tau_grid_expanded,
                                                    tau_grid = tau_grid,     
                                                    knot_points_grid = np.arange(0.1,1,0.1),
                                                    alpha_kappa = 5,
                                                    cov_mat_knots_init=cov_mat_knots_init,
                                                    A_matrices_G = A_matrices_G,
                                                  cov_matrices_G = cov_matrices_G,
                                                lambda_grid_log_prob= lambda_grid_log_prob,
                                                    beta_kappa = 1/3,
                                                    prior_on_t=prior_on_t,
                                                    base_quantile_mean=base_quantile_mean,
                                                    base_quantile_sd=base_quantile_sd,
                                                    base_quantile_v=base_quantile_v,
                                                    base_quantile_dist=base_quantile_dist)
        
        # Update phi_sample with new phis
        phi_sample[c_id-1] = phi_update
        eta1_sample[c_id-1] = eta_1_update
        eta2_sample[c_id-1] = eta_2_update
        
    return phi_sample, eta1_sample, eta2_sample


def sample_sigma_reg(sigma_reg_cur:np.array,
                     y:np.array,
                     x:np.array,
                     partition:np.array,
                     v_0:float,
                     sigma2_0:float,
               phi:np.array)->np.array:
    """ Samples phi from the full conditional, note this is for a linear regression problem

    Args:
        phi_cur (np.array): current value for phi
        y (float): n dim array of y values 
        x (np.array): n x p array of x values (note add 1s for intercept)
        partition (np.array): current partition
        phi_mean_prior (np.array): prior mean for coefficients
        phi_cov_prior (np.array): prior covariance for coefficients
        sigma_reg (float): error in linear regression outcome

    Returns:
        np.array: Sample from full conditional of phi
    """

    active_clust_ids = np.unique(partition)
    sigma_reg_sample = sigma_reg_cur.copy()

    
    # Update those in phi cur that are active clusters
    for c_id in active_clust_ids:

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

        sigma_reg_sample[c_id-1] = clust_sigma_sample
        
    return sigma_reg_sample



def calc_log_joint(partition:np.array, phi:np.array,
                   y:np.array, x:np.array,
                   sim_mat:np.array, order:np.array,
                   alpha:float, delta:float, sigma_reg:float) -> float:
    """ Calculates log joint of the EPA regression model.

    Args:
        partition (np.array): np.array (1xn) dim array of cluster indices for each data points
        phi (np.array): parameters of the sampling model
        y (np.array): y data
        x (np.array): X data
        sim_mat (np.array): n x n matrix of similarities 
        order (np.array): array of 0:(n-1) of any order indicating the (randomly sampled order)
        alpha (float): alpha parameter of distribution
        delta (float): delta parameter of distribution
        sigma_reg (float): sigma parameter of linear regression

    Returns:
        float: log joint
    """
    
    # Partition log prob
    partition_log_prob = partition_log_pdf_fast(partition,
                                        sim_mat, 
                                        order, 
                                        alpha, delta)
  
    # phi log prob
    sampling_log_prob = 0
    for i,c in enumerate(partition):
        sampling_log_prob += unit_log_likelihood_pdf(y[i],
                                                     x[i],
                                                     phi[c-1], model='Gaussian', sigma_reg=sigma_reg)
    
    
    return partition_log_prob + sampling_log_prob

   

# Gibbs optimization code
@numba.njit()
def partition_log_pdf_factors(start_id:int,
                              return_fac: bool=False,
                              partition: np.ndarray=np.array([]),
                      sim_mat: np.ndarray=np.array([[]]),
                      order: np.ndarray=np.array([]),
                      alpha: float=1,
                      delta: float=0) -> float:
    """ Calculates the log probability of with some factors pre-computed. Computes factors in the partition 
    from start_id onwards

    Args:
        start_id (int): Index from with to start computing the partial pdf, note this corresponds to 
        delta_{start_id}
        partition (np.ndarray): Cluster indices of each point
        sim_mat (np.ndarray): n x n matrix of similarities
        order (np.ndarray): n dim vector of order indicating the randomly sampled order
        alpha (float): alpha parameter
        delta (float): delta parameter

    Returns:
        float: log probability of partition
    """    

    assert len(order) == len(partition), "Length of order and partition inputs not the same"

    q = 1 # No of clusters in t-1 (? one subset in empty set?)
    
    
    log_p = np.zeros(len(partition))

    # Loop over each element (based on order) (think this is necessary)
    for t_1, o in enumerate(order):
        
        t = t_1 + 1 # to match t in formula

        c_o = partition[o]

        if t_1 == 0:
            points_seen = order[0:t_1]
        else:
            points_seen = order[0:t_1-1]
            
        clust_labels_seen = partition[points_seen]
        q = len(np.unique(clust_labels_seen))

        if c_o in clust_labels_seen:  # c_o belongs to existing cluster

            # Extract members of cluster c_o is in 
            cluster_member_position = np.where(clust_labels_seen==c_o)[0]
            cluster_members = order[cluster_member_position]

            # calculate p_t
            point_pairwise_dist = sim_mat[o,:]
            p_t = ((t-1 - delta * q)/(alpha + t-1)) * (np.sum(point_pairwise_dist[cluster_members]) / \
                                                        np.sum(point_pairwise_dist[points_seen]))


        else:  # c_o belongs to new cluster
            # calculate p_t
            p_t = (alpha + delta*q)/(alpha+t-1)

        log_p[t_1] = np.log(p_t)

    
    return log_p


@numba.njit()
def partition_log_pdf_partial(start_id:int,
                              return_fac: bool=False,
                              partition: np.ndarray=np.array([]),
                      sim_mat: np.ndarray=np.array([[]]),
                      order: np.ndarray=np.array([]),
                      alpha: float=1,
                      delta: float=0 ) -> float:
    """ Calculates the log probability of with some factors pre-computed. Computes factors in the partition 
    from start_id onwards

    Args:
        start_id (int): Index from with to start computing the partial pdf, note this corresponds to 
        delta_{start_id}
        partition (np.ndarray): Cluster indices of each point
        sim_mat (np.ndarray): n x n matrix of similarities
        order (np.ndarray): n dim vector of order indicating the randomly sampled order
        alpha (float): alpha parameter
        delta (float): delta parameter

    Returns:
        float: log probability of partition
    """    

    assert len(order) == len(partition), "Length of order and partition inputs not the same"

    q = 1 # No of clusters in t-1 (? one subset in empty set?)
    
    log_p = 0

    # Loop over each element (based on order) (think this is necessary)
    for t_1, o in enumerate(order[start_id:]):
    
        
        t = t_1 + 1 + start_id # to match t in formula

        c_o = partition[o]

        if t == 1:
            points_seen = order[0:t-1]
        else:
            points_seen = order[0:t-2]
            
        clust_labels_seen = partition[points_seen]
        q = len(np.unique(clust_labels_seen))

        if c_o in clust_labels_seen:  # c_o belongs to existing cluster

            # Extract members of cluster c_o is in 
            cluster_member_position = np.where(clust_labels_seen==c_o)[0]
            cluster_members = order[cluster_member_position]

            # calculate p_t
            point_pairwise_dist = sim_mat[o,:]
            p_t = ((t-1 - delta * q)/(alpha + t-1)) * (np.sum(point_pairwise_dist[cluster_members]) / \
                                                       np.sum(point_pairwise_dist[points_seen]))

        else:  # c_o belongs to new cluster
            # calculate p_t
            p_t = (alpha + delta*q)/(alpha+t-1)

        log_p += np.log(p_t)

    
    return log_p





@numba.njit(parallel=True)
def generate_candidate_partitions_alt_parallel(i:int,
                                      partition:np.array,
                                      names_used:List[int])->Tuple[List[np.array], List[int], int]:
    """generates candidate partitions but with new clust partitions having +1 of max partition
    note always puts the new clust partition as the first element (Parallel version)

    Args:
        i (int): which data point (in partition) to generate candidates for
        partition (np.array):  1 x n dim array of cluster indices for each data points
        names_used (List[int]): list of ids which have been used as cluster names so far

    Returns:
        Tuple(List[np.array], List[int], int): candidate_partitions: list of arrays where each array corresponds to candidate partition
                                               clust_ids: the unique cluster labels from the input partition
                                               new_clust_id: the unique cluster labels from the output partition
    """
    
    clust_ids = np.unique(partition)

    # generate label for new cluster as +1 of the largest current label
    new_clust_id = names_used.max() + 1

    # insert at 0 element, the new cluster id name
    clust_ids = np.concatenate((np.array((new_clust_id,)), clust_ids))
    
    # Make a matrix to store all the posisble candidate partitions
    candidate_partitions = np.zeros((len(clust_ids), len(partition)))

    # Loop over each unique index and generate a partition where the i-th entry's cluster id is replaced 
    for j in prange(len(clust_ids)):
        cand_part = partition.copy()
        cand_part[i] = clust_ids[j]

        candidate_partitions[j,:] = cand_part
    return candidate_partitions, clust_ids, new_clust_id


def generate_candidate_partitions_alt(i:int,
                                      partition:np.array,
                                      names_used:List[int])->Tuple[List[np.array], List[int], int]:
    """generates candidate partitions but with new clust partitions having +1 of max partition
    note always puts the new clust partition as the first element

    Args:
        i (int): which data point (in partition) to generate candidates for
        partition (np.array):  1 x n dim array of cluster indices for each data points
        names_used (List[int]): list of ids which have been used as cluster names so far

    Returns:
        Tuple(List[np.array], List[int], int): candidate_partitions: list of arrays where each array corresponds to candidate partition
                                               clust_ids: the unique cluster labels from the input partition
                                               new_clust_id: the unique cluster labels from the output partition
    """
    
    clust_ids = list(np.unique(partition))

    # generate label for new cluster as +1 of the largest current label
    new_clust_id = max(names_used) + 1

    # insert at 0 element, the new cluster id name
    clust_ids.insert(0,new_clust_id)
    candidate_partitions = []

    # Loop over each unique index and generate a partition where the i-th entry's cluster id is replaced 
    for c in clust_ids:
        cand_part = partition.copy()
        cand_part[i] = c

        candidate_partitions.append(cand_part)
    return candidate_partitions, clust_ids, new_clust_id


# Returns all factors and partition componenets of each candidate
@numba.njit()
def calc_partition_probs_return_factors(candidate_partitions,
                         clust_ids,
                         pre_compute_factors,
                        order_place, 
                        return_fac,
                        sim_mat, 
                        order, 
                        alpha, delta):
        
    pdf_partition_component = np.zeros(len(clust_ids))
    
    for j in range(len(clust_ids)):
        
        c_name = clust_ids[j]
        cp = candidate_partitions[j,:]
        
    
        partition_comp_partial =  partition_log_pdf_factors_parallel(order_place,  # start_id
                                                    return_fac,
                                                    cp,
                                                    sim_mat, 
                                                    order, 
                                                    alpha, delta) 

        partition_comp = partition_comp_partial.sum()
        
        pdf_partition_component[j] = partition_comp
    
    return pdf_partition_component, partition_comp_partial


# Returns only partition componenets of each candidate
@numba.njit(parallel=True)
def calc_partition_probs(candidate_partitions,
                         clust_ids,
                         pre_compute_factors,
                        order_place, 
                        return_fac,
                        sim_mat, 
                        order, 
                        alpha, delta):
        
    pdf_partition_component = np.zeros(len(clust_ids))
    
    for j in prange(len(clust_ids)):
        
        c_name = clust_ids[j]
        cp = candidate_partitions[j,:]
        

        partition_comp_partial =  partition_log_pdf_partial_parallel(order_place,  # start_id
                                                    False,
                                                    cp,
                                                    sim_mat, 
                                                    order, 
                                                    alpha, delta) 

        partition_comp = partition_comp_partial + np.sum(pre_compute_factors[:order_place])
            
        
        pdf_partition_component[j] = partition_comp
    
    return pdf_partition_component



def sample_conditional_i_clust_gibbs_opti_parallel(i:int,
                                          order_place:int,
                                          partition:np.array,
                                          return_fac:bool,
                                          pre_compute_factors:np.array,
                                          alpha:float,
                                          delta:float,
                                          sim_mat:np.array,
                                          order:np.array,
                                          phi:np.array,
                                          y:np.array,
                                          x:np.array, 
                                          sigma_reg:np.array,
                                          v_0:float,
                                          sigma2_0:float,
                                          names_used:np.array, 
                                          phi_base_mean:float,
                                          phi_base_cov:np.array,
                                          reordering:bool=False) -> Tuple[np.array, np.array, List[int]]:    
    """
    Parallel version

    partition: np.array (1xn) dim array of cluster indices for each data points
    i: which data point to generate candidates for
    order_place: where in the order the data point is.
    pre_compute_factors: np.array (1xn) dim array of factors calculated in the last term.
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
    candidate_partitions, clust_ids, new_clust_id = generate_candidate_partitions_alt_parallel(i,
                                                                                      partition,
                                                                                      names_used)
    updated_names = names_used.copy()
    #print('updated names: ', updated_names)
    # calc log likelihood of each partition
    log_liks = np.zeros(len(clust_ids))
    
    
    if return_fac:
        # Calc partition component
        pdf_partition_component, partition_comp_partial = calc_partition_probs_return_factors(candidate_partitions,
                                                         clust_ids,
                                                       pre_compute_factors,
                                                        order_place, 
                                                        return_fac,
                                                        sim_mat, 
                                                        order, 
                                                        alpha, delta)
    else:
        # Calc partition component
        pdf_partition_component = calc_partition_probs(candidate_partitions,
                                                         clust_ids,
                                                       pre_compute_factors,
                                                        order_place, 
                                                        return_fac,
                                                        sim_mat, 
                                                        order, 
                                                        alpha, delta)

    # Calc likelihood component
    for j in range(len(clust_ids)):
        
        c_name = int(clust_ids[j])
        cp = candidate_partitions[j,:]
        
        if c_name == new_clust_id: # if it is the new cluster (Note this is always first)

            phi_new = np.random.multivariate_normal(phi_base_mean,phi_base_cov) # sample something from prior
            
            gam_alpha_prior = v_0/2
            gam_beta_prior = (v_0*sigma2_0) /2
            sigma_reg_new = gamma.rvs(gam_alpha_prior, scale=1/gam_beta_prior)

            ll_comp = unit_log_likelihood_pdf(y_i,
                                    x_i, phi_new,
                                    model='Gaussian',
                                    sigma_reg=sigma_reg_new)

        else: # it is an existiing cluster
            #print("phi used: ", phi[c_name-1])
            ll_comp = unit_log_likelihood_pdf(y_i,
                                x_i, phi[c_name-1],  # 1 based indexing for cluster but 0 for phi
                                model='Gaussian',
                                sigma_reg=sigma_reg[c_name-1])
        
        # calculate probability of candidate partition and collect
        log_prob_cp = ll_comp
        #print(f"New LP {c_name}: {log_prob_cp}")
        #print(f'New Partition comp {c_name}: , {partition_comp}')

        #print("Partition: ", cp)
        #print(' ')
        log_liks[j] = log_prob_cp

    log_liks = log_liks + pdf_partition_component
    # Collect probabilities and normalize with log-sum-exp trick
    cand_probs = np.exp(log_liks - logsumexp(log_liks))

    # sample outcome partition for candidate partitions
    cand_part_choice = np.random.choice(np.arange(len(cand_probs)), p=cand_probs)
    output_partition = candidate_partitions[cand_part_choice]

    if cand_part_choice == 0:  # if new partition formed then update an extra name and phi
        #updated_names.append(new_clust_id)
        #clust_ids = np.insert(clust_ids, 0, new_clust_id)
        
        updated_names = np.insert(updated_names, len(updated_names), new_clust_id)
        phi = np.concatenate([phi,np.array([phi_new])])
        sigma_reg = np.concatenate([sigma_reg, np.array([sigma_reg_new])])
    
    
    if reordering == True:
        
    # Re-indexing step
        # re-index partition
        new_labels, existing_labels, relab_map = gen_reindex(output_partition)
        recoded_partition = remap_partition(output_partition, relab_map)

        # reset phi
        # Drop all empty elements
        phi_new = phi[existing_labels-1]
        sigam_reg_new = sigma_reg[existing_labels-1]
        #print(existing_labels-1)
        
        if return_fac:
            return recoded_partition, phi_new, np.sort(new_labels).astype('int'), partition_comp_partial, sigam_reg_new
        else:
            return recoded_partition, phi_new, np.sort(new_labels).astype('int'), sigam_reg_new

    
    else:
        
        if return_fac: 
            return output_partition, phi, updated_names, partition_comp_partial, sigma_reg
        else:
            return output_partition, phi,updated_names, sigma_reg





def sample_conditional_i_clust_gibbs_opti_parallel_QR(i:int,
                                          order_place:int,
                                          partition:np.array,
                                          return_fac:bool,
                                          pre_compute_factors:np.array,
                                          alpha:float,
                                          delta:float,
                                          sim_mat:np.array,
                                          order:np.array,
                                          phi:np.array,
                                           eta1_mat: np.array, 
                                            eta2_mat: np.array,
                                          y:np.array,
                                          x:np.array, 
                                        A_matrices_G: np.array,
                                        cov_matrices_G: np.array,
                                        lambda_grid_log_prob: np.array,
                                          w_cov_mat:np.array,
                                          alpha_kappa:float,
                                          beta_kappa:float,
                                          tau_grid:np.array,
                                          tau_grid_expanded:np.array,
                                         names_used:np.array,
                                        base_quantile_mean=0.0,
                                         base_quantile_sd=1.0,
                                         base_quantile_v=1.0,
                                         base_quantile_dist='norm',
                                          reordering:bool=True,
                                          splice=True,
                                          prior_on_t=False,
                                          multivariate_x=False,
                                          longi_x=False) -> Tuple[np.array, np.array, List[int]]:    
    """
    Parallel version and with QR Likelihood

    partition: np.array (1xn) dim array of cluster indices for each data points
    i: which data point to generate candidates for
    order_place: where in the order the data point is.
    pre_compute_factors: np.array (1xn) dim array of factors calculated in the last term.
    sim_mat: n x n matrix of similarities 
    phis: matrix / vector each column is beta_j
    order: array of 0:(n-1) of any order indicating the (randomly sampled order)
    alpha: alpha parameter of distribution
    delta: delta parameter of distribution
    y: y_data - Full data array of N points
    x: x_data - Full data array of N points / Nxp (if multivariate) / NxTxp (if longitudinal + multivar)
    names_used: str of cluster ids used thus far
    reordering: whether to re-order cluster labels to prevent cluster labels from getting large
    """
    
    y_i = y[i]
    x_i = x[i]  # Possibly p dim and possibly Txp dim if longitudinal

    if multivariate_x:

        if not longi_x:
            n_covar = x_i.shape[0]
            n_time_point = np.nan

        else:
            n_covar = x_i.shape[1]
            n_time_point = x_i.shape[0]

            # Convert x to cross-sectional rep
            x_cross_sec = x.reshape(-1,n_covar)  # Size N*T , p

        # Expand x_i to have correct shape
        #x_i = x_i.reshape(1,-1) # Add dimension
    else:
        n_covar = 1



    # generate candidate partitions
    candidate_partitions, clust_ids, new_clust_id = generate_candidate_partitions_alt_parallel(i,
                                                                                      partition,
                                                                                      names_used)
    updated_names = names_used.copy()
    #print('updated names: ', updated_names)
    # calc log likelihood of each partition
    log_liks = np.zeros(len(clust_ids))
    
    # On the first call to gibbs sampler we return factors
    if return_fac:
        # Calc partition component
        pdf_partition_component, partition_comp_partial = calc_partition_probs_return_factors(candidate_partitions,
                                                         clust_ids,
                                                       pre_compute_factors,
                                                        order_place, 
                                                        return_fac,
                                                        sim_mat, 
                                                        order, 
                                                        alpha, delta)
    else:
        # Calc partition component
        pdf_partition_component = calc_partition_probs(candidate_partitions,
                                                         clust_ids,
                                                       pre_compute_factors,
                                                        order_place, 
                                                        return_fac,
                                                        sim_mat, 
                                                        order, 
                                                        alpha, delta)
    
    # Calc likelihood component
    for j in range(len(clust_ids)):
        
        # Get cluster id and partition
        c_name = int(clust_ids[j])
        cp = candidate_partitions[j,:]
        
        if c_name == new_clust_id: # if it is the new cluster (Note this is always first)

            phi_new = phi_prior_sample(w_cov_mat,
                                       alpha_kappa,
                                       beta_kappa,
                                       prior_on_t=prior_on_t,
                                       multivariate_x=multivariate_x,
                                       n_covar=n_covar) # sample something from prior
            if multivariate_x:
                if prior_on_t:
                    m = (len(phi_new)-4 - 2*n_covar)//2
                else:
                    m = (len(phi_new)-3 - 2*n_covar)//2
            else:
                if prior_on_t:
                    m = (len(phi_new)-5)//2
                else:
                    m = (len(phi_new)-4)//2
            
            w1_knot_new = phi_new[0:m]
            w2_knot_new = phi_new[m:2*m]
            mu_new = phi_new[2*m]
            
            if multivariate_x:
                gamma_new = phi_new[2*m+1: 2*m+1+n_covar]
                sigma1_new = phi_new[2*m+1+n_covar] #np.exp(phi_current[2*m+2]) # TODO: Potential bug no need to exponentiate
                sigma2_new = phi_new[2*m+1+n_covar + 1] #np.exp(phi_current[2*m+3])
                
                if prior_on_t:
                    v_new = phi_new[2*m+1+n_covar + 2]
                    x_alpha_new = phi_new[2*m+1+n_covar + 3 : 2*m+1+n_covar + 3 + n_covar]  # Fix

                else:
                    v_new = base_quantile_v
                    x_alpha_new = phi_new[2*m+1+n_covar + 2 : 2*m+1+n_covar + 2 +n_covar]  # Fix


            else:
                gamma_new = phi_new[2*m+1]
                sigma1_new = phi_new[2*m+2] #np.exp(phi_current[2*m+2]) # TODO: Potential bug no need to exponentiate
                sigma2_new = phi_new[2*m+3] #np.exp(phi_current[2*m+3])
                
                if prior_on_t:
                    v_new = phi_new[2*m+4]
                else:
                    v_new = base_quantile_v

            # Project x if needed
            if multivariate_x:
                if not longi_x:
                    x_proj_new = project_x(x, x_alpha_new)
                    proj_x_new_i = x_proj_new[i]
                    if np.any(np.isnan(x_proj_new)):
                        print("NaN in Proj X New detected: ")
                        print('x_alpha_new: ', x_alpha_new)
                        print('x_proj: ', x_proj_new)
                else:
                    # use x_cross sectional to project
                    x_proj_cross_sec = project_x(x_cross_sec, x_alpha_new)  # produces (N*T),1 vector
                    # Reshape vector into longitudinal 
                    x_proj_longi = x_proj_cross_sec.reshape(-1,n_time_point)  # Produces (N,T) matrix
                    # Extract T points corresponding to xi
                    proj_x_new_i = x_proj_longi[i,:]  # T dim vector


            else:
                proj_x_new_i = None

            """
            gamma_new = phi_new[2*m+1]
            sigma1_new = phi_new[2*m+2]
            sigma2_new = phi_new[2*m+3]

            if prior_on_t:
                v_new = phi_new[2*m+4]
            else:
                v_new = base_quantile_v
            """
            
            # Get w1_approx and w2_approx from w1_knot and w2_knot
            # Update w1_sample
            w1_approx_prop_new, _ = calc_mixture_knot_approx_marginalized(w1_knot_new,
                                                                            a_kappa=alpha_kappa,
                                                                            b_kappa=beta_kappa,
                                                                            tau_grid=tau_grid_expanded,
                                                                            A_g_matrices=A_matrices_G,
                                                                            cov_mat_knot_store=cov_matrices_G,
                                                                            lambda_grid_log_prob=lambda_grid_log_prob)


            # Update w2_sample
            w2_approx_prop_new, _ = calc_mixture_knot_approx_marginalized(w2_knot_new,
                                                                            a_kappa=alpha_kappa,
                                                                            b_kappa=beta_kappa,
                                                                            tau_grid=tau_grid_expanded,
                                                                            A_g_matrices=A_matrices_G,
                                                                            cov_mat_knot_store=cov_matrices_G,
                                                                            lambda_grid_log_prob=lambda_grid_log_prob)

            # Get eta 1, eta 2 for component (Do approximation for w + logistic transform)
            eta1_new = eta_function_i_vector(tau_input=tau_grid,
                               w_vals=w1_approx_prop_new,
                               tau_grid=tau_grid_expanded,
                               mean=base_quantile_mean,
                               sd=base_quantile_sd,
                               v=v_new,
                               sigma=sigma1_new,
                               dist=base_quantile_dist)
            
            eta2_new = eta_function_i_vector(tau_input=tau_grid,
                               w_vals=w2_approx_prop_new,
                               tau_grid=tau_grid_expanded,
                               mean=base_quantile_mean,
                               sd=base_quantile_sd,
                               v=v_new,
                               sigma=sigma2_new,
                               dist=base_quantile_dist)

            if not longi_x:  # if not longi x then these are scalar values so convert back to matrix
                xi_input = np.array([x_i])
                yi_input = np.array([y_i])
            else: # if longi x then these are vectors
                xi_input = x_i
                yi_input = y_i

            # calculate addition to likelihood from the unit
            ll_comp = eval_ll_eta(y_vals_true=yi_input,
                                         x_vals=xi_input,
                                         eta_1=eta1_new,
                                         eta_2=eta2_new,
                                         sigma_1=sigma1_new,
                                         sigma_2=sigma2_new,
                                         tau_grid=tau_grid,
                                         tau_grid_expanded=tau_grid_expanded,
                                         mu=mu_new,
                                         gamma=gamma_new,
                                         base_quantile_mean=base_quantile_mean,
                                         base_quantile_sd=base_quantile_sd,
                                         base_quantile_v=v_new,
                                         base_quantile_dist=base_quantile_dist,
                                         splice=splice,
                                         multivariate_x=multivariate_x,
                                         proj_x=proj_x_new_i)
            if longi_x:
                ll_comp = ll_comp / 1#n_time_point

        else: # it is an existiing cluster
            #print("phi used: ", phi[c_name-1])
            
            # Extract phi
            #print('c_name: ',c_name)
            #print('c_name: ',c_name)
            #print('phi: ', phi.shape)
            phi_vec = phi[c_name-1,:]  # Extract w1_j, w2_j, mu_j, gamma_j, sigma1,sigma2
            
            w1_comp = phi_vec[0:m]
            w2_comp = phi_vec[m:2*m]
            mu_comp = phi_vec[2*m]


            mu_comp = phi_vec[2*m]

            if multivariate_x:
                gamma_comp = phi_vec[2*m+1: 2*m+1+n_covar]
                sigma1_comp = phi_vec[2*m+1+n_covar] #np.exp(phi_current[2*m+2])
                sigma2_comp = phi_vec[2*m+1+n_covar + 1] #np.exp(phi_current[2*m+3])
                
                if prior_on_t:
                    v_comp = phi_vec[2*m+1+n_covar + 2]
                    x_alpha_comp = phi_vec[2*m+1+n_covar + 3 : 2*m+1+n_covar + 3 + n_covar]  

                else:
                    v_comp = base_quantile_v
                    x_alpha_comp = phi_vec[2*m+1+n_covar + 2 : 2*m+1+n_covar + 2 +n_covar]  


            else:
                gamma_comp = phi_vec[2*m+1]
                sigma1_comp = phi_vec[2*m+2] #np.exp(phi_current[2*m+2]) 
                sigma2_comp = phi_vec[2*m+3] #np.exp(phi_current[2*m+3])
                
                if prior_on_t:
                    v_comp = phi_vec[2*m+4]
                else:
                    v_comp = base_quantile_v

            """
            if multivariate_x:
                x_proj_comp = project_x(x, x_alpha_comp)
                proj_x_comp_i = x_proj_comp[i]
                if np.any(np.isnan(x_proj_new)):
                    print("NaN in Proj X Comp detected: ")
                    print('x_alpha_comp: ', x_alpha_comp)
                    print('x_proj_comp: ', x_proj_comp)
            else:
                proj_x_comp_i = None
            """

            if multivariate_x:
                if not longi_x:
                    x_proj_comp = project_x(x, x_alpha_comp)
                    proj_x_comp_i = x_proj_comp[i]
                    if np.any(np.isnan(x_proj_comp)):
                        print("NaN in Proj X Comp detected: ")
                        print('x_alpha_comp: ', x_alpha_comp)
                        print('x_proj_comp: ', x_proj_comp)
                else:
                    # use x_cross sectional to project
                    x_proj_cross_sec_comp = project_x(x_cross_sec, x_alpha_comp)  # produces (N*T),1 vector
                    # Reshape vector into longitudinal to easily extract i-th entries
                    x_proj_longi_comp = x_proj_cross_sec_comp.reshape(-1,n_time_point)  # Produces (N,T) matrix
                    # Extract T points corresponding to xi
                    proj_x_comp_i = x_proj_longi_comp[i,:]  # T dim vector
            
            else:
                proj_x_comp_i = None


            # Extract etas
            eta_1_comp = eta1_mat[c_name-1,:]
            eta_2_comp = eta2_mat[c_name-1,:]
            
            if not longi_x:  # if not longi x then these are scalar values so convert back to matrix
                xi_input = np.array([x_i])
                yi_input = np.array([y_i])
            else: # if longi x then these are vectors
                xi_input = x_i
                yi_input = y_i

            ll_comp = eval_ll_eta(y_vals_true=yi_input,
                                         x_vals=xi_input,
                                         eta_1=eta_1_comp,
                                         eta_2=eta_2_comp,
                                         sigma_1=sigma1_comp,
                                         sigma_2=sigma2_comp,
                                         tau_grid=tau_grid,
                                         tau_grid_expanded=tau_grid_expanded,
                                         mu=mu_comp,
                                         gamma=gamma_comp,
                                         base_quantile_mean=base_quantile_mean,
                                         base_quantile_sd=base_quantile_sd,
                                         base_quantile_v=v_comp,
                                         base_quantile_dist=base_quantile_dist,
                                         splice=splice,
                                         multivariate_x=multivariate_x,
                                         proj_x=proj_x_comp_i)
            if longi_x:
                ll_comp = ll_comp / 1#n_time_point
        
        # calculate probability of candidate partition and collect
        log_prob_cp = ll_comp
        #print(f"New LP {c_name}: {log_prob_cp}")
        #print(f'New Partition comp {c_name}: , {partition_comp}')

        #print("Partition: ", cp)
        #print(' ')
        log_liks[j] = log_prob_cp
    #print("QR Log Likes: ", log_liks)
    #print('Partion Component: ',pdf_partition_component)
    
    if np.all(log_liks == -np.inf): # All QRs fail
        log_liks = np.array(len(log_liks)*[0])  # Just dont add anything and rely on EPA instead
    log_liks = log_liks + pdf_partition_component
    #print(ll_comp, pdf_partition_component)

    # Collect probabilities and normalize with log-sum-exp trick
    cand_probs = np.exp(log_liks - logsumexp(log_liks))
    #print('Candidate Probs: ',cand_probs)
    
    # sample outcome partition for candidate partitions
    cand_part_choice = np.random.choice(np.arange(len(cand_probs)), p=cand_probs)
    output_partition = candidate_partitions[cand_part_choice]

    #print("Partition Sizes: ",np.bincount(output_partition.astype('int'))[1:])
    
    if cand_part_choice == 0:  # if new partition formed then update an extra name and phi
        #updated_names.append(new_clust_id)
        #clust_ids = np.insert(clust_ids, 0, new_clust_id)
        
        updated_names = np.insert(updated_names, len(updated_names), new_clust_id)
        phi = np.vstack([phi,phi_new])
        eta1_mat = np.vstack([eta1_mat,eta1_new])
        eta2_mat = np.vstack([eta2_mat,eta2_new])
            
    
    if reordering == True:
        
    # Re-indexing step
        # re-index partition
        new_labels, existing_labels, relab_map = gen_reindex(output_partition)
        recoded_partition = remap_partition(output_partition, relab_map)

        # reset phi
        # Drop all empty elements
        phi_new = phi[existing_labels-1]
        eta1_mat_new =  eta1_mat[existing_labels-1]
        eta2_mat_new =  eta2_mat[existing_labels-1]
                
        if return_fac:
            return recoded_partition, phi_new, np.sort(new_labels).astype('int'), partition_comp_partial, eta1_mat_new,eta2_mat_new
        else:
            return recoded_partition, phi_new, np.sort(new_labels).astype('int'), eta1_mat_new,eta2_mat_new

    
    else:
        
        if return_fac: 
            return output_partition, phi, updated_names, partition_comp_partial, eta1_mat, eta2_mat
        else:
            return output_partition, phi, updated_names, eta1_mat, eta2_mat



@numba.njit(parallel=True)
def partition_log_pdf_partial_parallel(start_id:int,
                              return_fac: bool=False,
                              partition: np.ndarray=np.array([]),
                      sim_mat: np.ndarray=np.array([[]]),
                      order: np.ndarray=np.array([]),
                      alpha: float=1,
                      delta: float=0 ) -> float:
    """ Calculates the log probability of with some factors pre-computed. Computes factors in the partition 
    from start_id onwards

    Args:
        start_id (int): Index from with to start computing the partial pdf, note this corresponds to 
        delta_{start_id}
        partition (np.ndarray): Cluster indices of each point
        sim_mat (np.ndarray): n x n matrix of similarities
        order (np.ndarray): n dim vector of order indicating the randomly sampled order
        alpha (float): alpha parameter
        delta (float): delta parameter

    Returns:
        float: log probability of partition
    """    

    assert len(order) == len(partition), "Length of order and partition inputs not the same"

    q = 1 # No of clusters in t-1 (? one subset in empty set?)
    
    log_p = np.zeros(len(order))
    #t_1 = 0
    indexes = np.arange(-start_id,len(order)+start_id)
    
    # Loop over each element (based on order) (think this is necessary)
    for j in prange(start_id,len(order)):
        #print(j)
        o = order[j]
        t_1 = indexes[j]
        
        t = t_1 + 1 + start_id # to match t in formula

        c_o = partition[o]

        if t == 1:
            points_seen = order[0:t-1]
        else:
            points_seen = order[0:t-2]
            
        clust_labels_seen = partition[points_seen]
        q = len(np.unique(clust_labels_seen))

        if c_o in clust_labels_seen:  # c_o belongs to existing cluster

            # Extract members of cluster c_o is in 
            cluster_member_position = np.where(clust_labels_seen==c_o)[0]
            cluster_members = order[cluster_member_position]

            # calculate p_t
            point_pairwise_dist = sim_mat[o,:]
            p_t = ((t-1 - delta * q)/(alpha + t-1)) * (np.sum(point_pairwise_dist[cluster_members]) / \
                                                       np.sum(point_pairwise_dist[points_seen]))

        else:  # c_o belongs to new cluster
            # calculate p_t
            p_t = (alpha + delta*q)/(alpha+t-1)

        #log_p += np.log(p_t)
        log_p[j] = np.log(p_t)
        #t_1 += 1
    
    return log_p.sum()


@numba.njit(parallel=True)
def partition_log_pdf_factors_parallel(start_id:int,
                              return_fac: bool=False,
                              partition: np.ndarray=np.array([]),
                      sim_mat: np.ndarray=np.array([[]]),
                      order: np.ndarray=np.array([]),
                      alpha: float=1,
                      delta: float=0 ) -> float:
    """ Calculates the log probability of with some factors pre-computed. Computes factors in the partition 
    from start_id onwards

    Args:
        start_id (int): Index from with to start computing the partial pdf, note this corresponds to 
        delta_{start_id}
        partition (np.ndarray): Cluster indices of each point
        sim_mat (np.ndarray): n x n matrix of similarities
        order (np.ndarray): n dim vector of order indicating the randomly sampled order
        alpha (float): alpha parameter
        delta (float): delta parameter

    Returns:
        float: log probability of partition
    """    

    assert len(order) == len(partition), "Length of order and partition inputs not the same"

    q = 1 # No of clusters in t-1 (? one subset in empty set?)
    
    log_p = np.zeros(len(order))
    #t_1 = 0
    indexes = np.arange(len(order))
    
    # Loop over each element (based on order) (think this is necessary)
    for j in prange(len(order)):
        #print(j)
        o = order[j]
        t_1 = indexes[j]
        
        t = t_1 + 1 # to match t in formula

        c_o = partition[o]

        if t == 1:
            points_seen = order[0:t-1]
        else:
            points_seen = order[0:t-2]
            
        clust_labels_seen = partition[points_seen]
        q = len(np.unique(clust_labels_seen))

        if c_o in clust_labels_seen:  # c_o belongs to existing cluster

            # Extract members of cluster c_o is in 
            cluster_member_position = np.where(clust_labels_seen==c_o)[0]
            cluster_members = order[cluster_member_position]

            # calculate p_t
            point_pairwise_dist = sim_mat[o,:]
            p_t = ((t-1 - delta * q)/(alpha + t-1)) * (np.sum(point_pairwise_dist[cluster_members]) / \
                                                       np.sum(point_pairwise_dist[points_seen]))

        else:  # c_o belongs to new cluster
            # calculate p_t
            p_t = (alpha + delta*q)/(alpha+t-1)

        #log_p += np.log(p_t)
        log_p[j] = np.log(p_t)
        #t_1 += 1
    
    return log_p
