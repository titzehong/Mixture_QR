# Functions to for GP Approximation
import numpy as np
import numba
from sklearn.metrics import pairwise_distances
from scipy.stats import norm, t, gamma, uniform, beta
from numba_stats import norm as numba_norm
from numba import prange
import scipy as sp
from typing import List
from  scipy.special import gammaln
from math import gamma as gamma_fn



#############################################################
#### 2007 Tokdar Paper GP Approx + Mixture T correlated ####
############################################################
# GPs here have option for rho, kappa and lambda (even though kappa and lambda can be marginalized)
@numba.njit
def covariance_function_single_var(in_1: np.ndarray,
                                   in_2: np.ndarray,
                                   kappa: float,
                                   rho: float,
                                   lambd:float,
                                   with_kappa:float = True) -> float:
    
    """ GP Covariance Function for correlated GP, single input

    Args:
        in_1 (np.array): First input
        in_2 (np.array): Second input
        kappa (float): kappa value
        rho (float): rho value
        lambd (float): lambd value

    Returns:
        _type_: Covariance
    """    

    c_mat = np.array([[1,rho],
                      [rho,1]])
    
    # Use x or condition to get this .....
    i_1 = int(in_1[0])
    i_2 = int(in_2[0])
    
    tau_1 = in_1[1]
    tau_2 = in_2[1]
    
    if with_kappa:
        return (kappa**2)*(c_mat[i_1,i_2])*np.exp(-(lambd**2)*(tau_1-tau_2)**2)
    else:
        return (c_mat[i_1,i_2])*np.exp(-(lambd**2)*(tau_1-tau_2)**2)


def covariance_function_single_var_vector(knot_points: np.ndarray,
                                            input_tau: np.ndarray,
                                            kappa: float,
                                            rho: float,
                                            lambd:float,
                                            with_kappa:bool=True):
    """ GP Covariance Function For correlated GPs 

    Args:
        knot_points (np.array): M x 2 matrix with 1st-column being i index and 2nd column tau values
        input_tau (np.array): t x 2 matrix with 1st-column being i index and 2nd column tau values of desired grid
        kappa (float): kappa value
        rho (float): rho value
        lambd (float): lambd value

    Returns:
        _type_: Covariance
    """    

    tau_diffs = np.subtract.outer(knot_points[:,1], input_tau[:,1])
    rho_select = np.not_equal.outer(knot_points[:,0], input_tau[:,0])

    if with_kappa:
        return (kappa**2)*(1+rho*rho_select - 1*rho_select)*np.exp(-(lambd**2)*(tau_diffs)**2)
    else:
        return (1+rho*rho_select - 1*rho_select)*np.exp(-(lambd**2)*(tau_diffs)**2)


def covariance_mat_single_var(gp_input: np.ndarray,
                   kappa:float,
                   rho:float,
                   lambd:float,
                   with_kappa:bool=True) -> np.ndarray:
    
    """ Helper function to create a covariance matrix given input for correlated GP

    gp_input (np.array): Array of points to calculate cov matrix, n_samples x 2
    kappa (float): kappa value
    rho (float): rho value
    lambd (float): lambd value

    Returns:
        _type_: Covariance matrix
    """    
    
    metric_args = {'kappa': kappa,
                   'rho': rho,
                   'lambd': lambd,
                    'with_kappa': with_kappa}
    
    output_mat = pairwise_distances(gp_input, metric=covariance_function_single_var, **metric_args)
            
    return output_mat


def calc_knot_approx_v2(tau_in,
                        knot_points_t,
                        cov_mat_knots,
                        w_knot_points,
                        kappa,
                        rho,
                        lambd,
                        with_kappa:bool=True):
    """ Calculating GP conditional Mean approximation Without marginalizing kappa and with correlation term rho

    Args:
        tau_in (np.ndarray): Points at which to generate the approximation [L x 2], first column index 0 or 1 and second column is t value
        knot_points_t (np.ndarray): M dim 1d vector cotaining tau values at which w is known (assumes same tau for gp 1 and gp 2)
        cov_mat_knots (np.ndarray): (2*M, 2*M) dim matrix, the implied covariance matrix of known w points
        w_knot_points (np.ndarray): 2*M dim 1 dim vector containing values of w
        kappa (float): kappa value
        rho (float): rho value
        lambd (float): lambda value
        with_kappa (bool, optional): Whether to use kappa in calculation (Set to false if marginalized out). Defaults to True.
    Returns:
        _type_: _description_
    """
    # Uses W evaluated at knot points
    m = len(knot_points_t)
    
    # Calc covariance matrix 
    knot_sub_ids = np.array(m*[0] + m*[1]).reshape(-1,1)
    knot_points_t = np.concatenate([knot_points_t,
                                knot_points_t]).reshape(-1,1)

    knot_points = np.hstack([knot_sub_ids, knot_points_t])


     # Calculate (M*2, L) dim matrix comparing wanted points and current knot points 
    cov_input_knots = covariance_function_single_var_vector(knot_points,
                                            tau_in,
                                            kappa,
                                            rho,
                                            lambd,
                                            with_kappa=with_kappa)
    
    
    f_w_approx = w_knot_points @ np.linalg.inv(cov_mat_knots) @ cov_input_knots


    return f_w_approx


#### Mixture t - GP Approx Correlated ####
def precompute_approx_correlated(tau_grid: np.ndarray,
                      knot_points_grid: np.ndarray,
                      rho: float,
                      lambda_grid: np.ndarray):
    
    # Make the GP input correct
    m = len(knot_points_grid)
    knot_sub_ids = np.array(m*[0] + m*[1]).reshape(-1,1)
    knot_points_t = np.concatenate([knot_points_grid,
                                knot_points_grid]).reshape(-1,1)
    knot_gp_input = np.hstack([knot_sub_ids, knot_points_t])

    # Pre-compute things
    cov_mat_knot_store = []
    A_g_matrices = []
    G = len(lambda_grid)

    # Pre compute stuff
    for g in range(G):
        lambda_g = lambda_grid[g]
        
        # Compute c** (m x m)
        cov_mat_knots_g = covariance_mat_single_var(knot_gp_input,
                                                    kappa=0,
                                                    lambd=lambda_g,
                                                    rho=rho,
                                                    with_kappa=False)
        
        cov_mat_knot_store.append(cov_mat_knots_g)
        
        # compute C_0* (L x m)
        C_0 = np.zeros([len(tau_grid), len(knot_gp_input)])
        for l in range(len(tau_grid)):
            for m in range(len(knot_gp_input)):
                tau_l = tau_grid[l]
                tau_m = knot_gp_input[m]
                C_0[l,m] = covariance_function_single_var(tau_l,
                                                    tau_m,
                                                    kappa=0, 
                                                    rho=rho,
                                                    lambd = lambda_g,
                                                      with_kappa=False)

        cov_mat_knots_inv = np.linalg.inv(cov_mat_knots_g)
        A_g = C_0 @ cov_mat_knots_inv
        A_g_matrices.append(A_g)

    return np.array(cov_mat_knot_store), np.array(A_g_matrices) 

@numba.njit
def calc_mixture_knot_approx_marginalized_correlated(w_j_star: np.ndarray,
                                 a_kappa: float,
                                 b_kappa: float,
                                 tau_grid: np.ndarray,
                                 A_g_matrices: np.ndarray,
                                 cov_mat_knot_store: np.ndarray,
                                 lambda_grid_log_prob: np.ndarray):
    
    # Calc mixture
    G = len(A_g_matrices)
    M = len(w_j_star)

    norm_pdf_lambda = renorm_dist(lambda_grid_log_prob)

    # Have to do for all calculations
    t_log_pdfs = np.zeros(G)
    output_vec = np.zeros(len(tau_grid))
    matrix_term = np.zeros((A_g_matrices[0].shape[0], A_g_matrices[0].shape[1]))
    marginal_log_prob = 0
    comp_log_prob_store = np.zeros(G)
    for g in range(G):
        t_log_pdf = logpdf_t_numba(w_j_star,
                            np.zeros(M),
                            (b_kappa/a_kappa)*cov_mat_knot_store[g],
                            2*a_kappa)

        t_log_pdfs[g] = t_log_pdf
        comp_log_prob_store[g] = np.log(norm_pdf_lambda[g]) + t_log_pdf


    mixture_weights = renorm_dist(comp_log_prob_store)
    marginal_log_prob = np.log(sum(np.exp(comp_log_prob_store)))

    
    mixture_weights = mixture_weights.reshape(G,1,1)

    matrix_term = np.sum(mixture_weights*A_g_matrices,0)
    approx_vec = matrix_term @ w_j_star
    
    return approx_vec, marginal_log_prob

def calc_lambd_grid_corr(knot_gp_input,
                         kappa,
                         rho,
                          h=0.1,
                          rho_init=0.95,
                          with_kappa=False):
    
    end_rho_val = 1-rho_init

    # Start at rho=0.9
    lamb_init = np.sqrt( np.log(rho_init) / -(h**2) )

    lambd_collect = []
    lambd_collect.append(lamb_init)
    current_rho = np.exp(-(h**2)*lamb_init**2)
    current_cov = covariance_mat_single_var(knot_gp_input,
                                            kappa=kappa,
                                            lambd=lamb_init,
                                            rho=rho,
                                            with_kappa=with_kappa)
    lambd_prop = lamb_init + 0.01
    prop_cov = covariance_mat_single_var(knot_gp_input,
                                            kappa=kappa,
                                            lambd=lambd_prop,
                                            rho=rho,
                                        with_kappa=with_kappa)
                                            
    while current_rho > end_rho_val:
        
        counter = 0
        while d_kl(current_cov, prop_cov) < 1:
            lambd_prop += 0.005
            prop_cov = covariance_mat_single_var(knot_gp_input,
                                            kappa=kappa,
                                            lambd=lambd_prop,
                                                rho=rho,
                                                with_kappa=with_kappa)
            
            prop_rho = np.exp(-(h**2)*lambd_prop**2)
            
            if prop_rho < end_rho_val:
                break
            
            counter += 1
            if counter > 5000:
                print("Counter Max Reached")
                break
                
        current_cov = prop_cov
        current_lamb = lambd_prop
        current_rho = np.exp(-(h**2)*current_lamb**2)
        
        lambd_collect.append(current_lamb)
        
        lambd_prop = current_lamb+0.01

    return lambd_collect


############################################
#### Mixture t - GP Approx Uncorrelated ####
############################################
# Similar to 2012 paper version
@numba.njit
def covariance_gp_uncorr(tau_1: np.ndarray,
                        tau_2: np.ndarray,
                        lambd:float) -> float:
    
    """ GP Covariance Function for correlated GP, single input

    Args:
        tau_1 (float): First input
        tau_2 (float): Second input
        lambd (float): lambd value

    Returns:
        _type_: Covariance
    """    
    
    return np.exp(-(lambd**2)*(tau_1-tau_2)**2)

def covariance_matrix_gp_uncorr(gp_input: np.ndarray,
                   lambd:float) -> np.ndarray:
    
    """ Helper function to create a covariance matrix given input for correlated GP

    gp_input (np.array): Array of points to calculate cov matrix, n_tau x 1
    lambd (float): lambd value

    Returns:
        _type_: Covariance matrix
    """    
    
    metric_args = {'lambd': lambd}
    output_mat = pairwise_distances(gp_input, metric=covariance_gp_uncorr, **metric_args)
            
    return output_mat


def covariance_outer_product_gp_uncorr(knot_points: np.ndarray,
                                            input_tau: np.ndarray,
                                            lambd:float):
    """ Calculates covariance between knot points and input tau, producing (M x L) vector, M is size of knot points, L is size input points 

    Args:
        knot_points (np.array): M x 1 matrix at knot points tau values
        input_tau (np.array): t x 1 matrix with input points to be evaluated
        lambd (float): lambd value

    Returns:
        _type_: Covariance
    """    

    tau_diffs = np.subtract.outer(knot_points, input_tau)

    return np.exp(-(lambd**2)*(tau_diffs)**2)


def d_kl(cov1, cov2):
    """ Calculates KL div between two 0 mean normals with cov 1 and cov 2 respectivetly

    Args:
        cov1 (np.ndarray): Covariance Matrix of 1st Normal
        cov2 (np.ndarray): Covariance Matrix of 2nd Normal

    Returns:
        _type_: KL Div
    """
    return 0.5*(calc_logdet(cov2) - calc_logdet(cov1) - cov1.shape[0] + np.trace(np.linalg.inv(cov2)@cov1))

def lamb_pdf(lambd,
             a_beta=6,
             b_beta=4,
             h=0.1):
    """ Calculates implied PDF of lambda

    Args:
        lambd (lambda): input lambda to calculate the PDF 
        a_beta (int, optional): a parameter of beta distribution on rho. Defaults to 6.
        b_beta (int, optional): b parameter of beta distribution on rho. Defaults to 4.
        h (float, optional): h in specification of lambda distribution. Defaults to 0.1.

    Returns:
        _type_: pdf value of lambda
    """
    # Calculate implied rho from the given lambda value
    implied_rho = np.exp(-(h**2)*(lambd**2))

    # Get pdf value
    pdf_term = beta.pdf(implied_rho, a_beta, b_beta) 

    # Calculate value of jacobian
    jacobian_term = 1/np.abs(0.5*((np.log(implied_rho)/(-(h**2)))**(-0.5))*(1/(-(h**2)*implied_rho)))
    
    # Calculate implied PDF value
    pdf_val = pdf_term*jacobian_term
    
    return pdf_val

def calc_logdet(mat):
    
    vals, vecs = np.linalg.eigh(mat)
    logdet  = np.log(vals).sum()
    return logdet

def calc_lambd_grid_uncorr(knot_gp_input,
                    h=0.1,
                    rho_lambd_init=0.99):
    
    end_rho = 1-rho_lambd_init
    lamb_init = np.sqrt( np.log(rho_lambd_init) / -(h**2) )

    lambd_collect = []
    lambd_collect.append(lamb_init)
    current_rho = np.exp(-(h**2)*lamb_init**2)

    current_cov = covariance_matrix_gp_uncorr(knot_gp_input,
                                            lambd=lamb_init)
    lambd_prop = lamb_init + 0.01
    prop_cov = covariance_matrix_gp_uncorr(knot_gp_input,
                                            lambd=lambd_prop)
                                            
    while current_rho > end_rho:
        
        counter = 0
        while d_kl(current_cov, prop_cov) < 1:
            lambd_prop += 0.005
            prop_cov = covariance_matrix_gp_uncorr(knot_gp_input,
                                            lambd=lambd_prop)
            
            prop_rho = np.exp(-(h**2)*lambd_prop**2)
            
            if prop_rho < end_rho:
                break
            
            counter += 1
            if counter > 5000:
                break
                
        current_cov = prop_cov
        current_lamb = lambd_prop
        current_rho = np.exp(-(h**2)*current_lamb**2)
        
        lambd_collect.append(current_lamb)
        
        lambd_prop = current_lamb+0.01

    return lambd_collect

@numba.njit
def logsumexp(x):
    # Utility to do renormalization properly
    # https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

@numba.njit
def renorm_dist(log_p_vec):
    return np.exp(log_p_vec - logsumexp(log_p_vec))



def precompute_approx_uncorr(tau_grid: np.ndarray,
                      knot_points_grid: np.ndarray,
                      lambda_grid: np.ndarray):
    """ Precomputes covariance matrices and A matrices in the mixture approximation

    Args:
        tau_grid (np.ndarray): L x 1 vector of for 
        knot_points_grid (np.ndarray): _description_
        lambda_grid (np.ndarray): _description_

    Returns:
        _type_: _description_
    """

    # Pre-compute things
    cov_mat_knot_store = []
    A_g_matrices = []
    G = len(lambda_grid)

    # Pre compute stuff
    for g in range(G):
        lambda_g = lambda_grid[g]
        
        # Compute c** (m x m)
        cov_mat_knots_g = covariance_matrix_gp_uncorr(knot_points_grid,
                                                    lambd=lambda_g)
        
        cov_mat_knot_store.append(cov_mat_knots_g)
        
        # compute C_0* (L x m)
        C_0 = np.zeros([len(tau_grid), len(knot_points_grid)])
        for l in range(len(tau_grid)):
            for m in range(len(knot_points_grid)):
                tau_l = tau_grid[l]
                tau_m = knot_points_grid[m]
                C_0[l,m] = covariance_gp_uncorr(tau_l,
                                                tau_m,
                                                lambd = lambda_g)

        cov_mat_knots_inv = np.linalg.inv(cov_mat_knots_g)
        A_g = C_0 @ cov_mat_knots_inv
        A_g_matrices.append(A_g)

    return np.array(cov_mat_knot_store), np.array(A_g_matrices) 

@numba.njit
def calc_mixture_knot_approx_marginalized(w_j_star: np.ndarray,
                                 a_kappa: float,
                                 b_kappa: float,
                                 tau_grid: np.ndarray,
                                 A_g_matrices: np.ndarray,
                                 cov_mat_knot_store: np.ndarray,
                                 lambda_grid_log_prob: np.ndarray):
    
    # Calc mixture
    G = len(A_g_matrices)
    M = len(w_j_star)

    norm_pdf_lambda = renorm_dist(lambda_grid_log_prob)

    # Have to do for all calculations
    t_log_pdfs = np.zeros(G)
    for g in range(G):
        t_log_pdf = logpdf_t_numba(w_j_star,
                            np.zeros(M),
                            (b_kappa/a_kappa)*cov_mat_knot_store[g],
                            2*a_kappa)

        t_log_pdfs[g] = t_log_pdf

    matrix_term = np.zeros((A_g_matrices[0].shape[0], A_g_matrices[0].shape[1]))
    marginal_log_prob = 0
    comp_log_prob_store = np.zeros(G)

    # Calculate Approx
    for g in range(G):

        comp_log_prob_store[g] = np.log(norm_pdf_lambda[g]) + t_log_pdfs[g]

    mixture_weights = renorm_dist(comp_log_prob_store)
    marginal_log_prob = np.log(sum(np.exp(comp_log_prob_store)))
    for g in range(G):

        matrix_term += mixture_weights[g] * (A_g_matrices[g])


    approx_vec = matrix_term @ w_j_star
    
    return approx_vec, marginal_log_prob


def logpdf_t(x, mean, shape, df):
    # https://gregorygundersen.com/blog/2020/01/20/multivariate-t/
    dim = mean.size

    vals, vecs = np.linalg.eigh(shape)
    logdet     = np.log(vals).sum()
    valsinv    = np.array([1./v for v in vals])
    U          = vecs * np.sqrt(valsinv)
    dev        = x - mean
    maha       = np.square(np.dot(dev, U)).sum(axis=-1)

    t = 0.5 * (df + dim)
    A = gammaln(t)
    B = gammaln(0.5 * df)
    C = dim/2. * np.log(df * np.pi)
    D = 0.5 * logdet
    E = -t * np.log(1 + (1./df) * maha)

    return A - B - C - D + E

@numba.njit
def logpdf_t_numba(x, mean, shape, df):
    # https://gregorygundersen.com/blog/2020/01/20/multivariate-t/
    dim = mean.size

    vals, vecs = np.linalg.eigh(shape)
    logdet     = np.log(vals).sum()
    valsinv    = np.array([1./v for v in vals])
    U          = vecs * np.sqrt(valsinv)
    dev        = x - mean
    maha       = np.square(np.dot(dev, U)).sum(axis=-1)

    t = 0.5 * (df + dim)
    A = np.log(gamma_fn(t))
    B = np.log(gamma_fn(0.5 * df))
    C = dim/2. * np.log(df * np.pi)
    D = 0.5 * logdet
    E = -t * np.log(1 + (1./df) * maha)

    return A - B - C - D + E