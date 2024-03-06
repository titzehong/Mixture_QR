import numpy as np
import matplotlib.pyplot as plt
import numba 
from sklearn.metrics import pairwise_distances
from scipy.stats import norm, t, gamma, uniform, multivariate_t
from typing import Union
import scipy as sp
from scipy.stats import norm
from numba_stats import norm as numba_norm
from numba_stats import t as numba_t
from numba import prange
from typing import List
from   scipy.special import gammaln
from scipy.stats import t as student_t_univar

from GP_Approx import *



@numba.njit
def calc_grid_trapezoidal(tau_grid: np.ndarray, c_vals:np.ndarray, last_id: int) -> float:    
    """ Calculates trapezoidal approximation for a given grid and values

    Args:
        taus (np.ndarray): Grid of tau values
        c_vals (np.ndarray): Function evaluation at each point in taus
        last_id (int): the index of point in tau grid to numerically integrate until

    Returns:
        float: _description_
    """
    trap_lens = tau_grid[1:] - tau_grid[0:-1]
    trap_heights = c_vals[0:-1] + c_vals[1:]
    
    trapz_vols = (0.5 * trap_lens * trap_heights)
    
    return trapz_vols

@numba.njit
def get_interval(tau_in: Union[float, np.ndarray] ,
                 tau_grid: np.ndarray) -> int:
    """ Finds the upper index of the interval in tau_grid where tau_in is located

    Args:
        tau_in (float): input tau value
        tau_grid (np.ndarray): Grid of tau values

    Returns:
        int: t_l if t \\in [tau_grid[t_l-1], tau_grid[t_l]]
    """
        
    trapz_len = tau_grid[1] - tau_grid[0]

    return np.ceil((tau_in - tau_grid[0]) / trapz_len)

@numba.njit
def calculate_contiguous_row_sums_numba(in_matrix,
                                        start_column_indices,
                                        end_column_indices):
    row_sums = np.zeros(len(in_matrix))
    
    for i in prange(len(in_matrix)):
        start = start_column_indices[i]
        end = end_column_indices[i]
        row_sums[i] = sum(in_matrix[i, start:end])
    
    return row_sums



@numba.njit
def calc_grid_trapezoidal_vector(tau_grid,
                                  c_vals,
                                  c_samp_repeat,
                                 last_ids):
    
    #print('as')

    trapz_len = tau_grid[1] - tau_grid[0]

    #c_samp_repeat = np.repeat(c_vals[:,np.newaxis],len(tau_grid), axis=1).T
    
    
    mid_vals = calculate_contiguous_row_sums_numba(c_samp_repeat,
                                                   np.ones(len(last_ids), dtype='int'),
                                                   last_ids)

    trapz_sum = (trapz_len/2)*(c_vals[0] + 2*mid_vals + c_vals[last_ids])
    return trapz_sum
    
@numba.njit
def get_interval_vector(tau_in: Union[float, np.ndarray],
                 taus: np.ndarray) -> np.ndarray:
    
    #tau_max = len(taus)
    #out_mat = np.empty_like(tau_in,dtype='int')
    """
    trapz_len = taus[1] - taus[0]
    out_mat = np.ceil((tau_in - taus[0]) / trapz_len)
    out_mat = out_mat.astype(np.int32)
    """

    out_mat = np.searchsorted(taus, tau_in) 
    
    return out_mat


def logistic_transform(tau_input: float,
                       tau_grid: np.ndarray,
                       c_vals_i: np.ndarray)->float:
    """ Calculate the logistic transform of xi for a given tau and zeta function

    Args:
        tau_input (float): Input tau value
        tau_grid (np.ndarray): Grid of tau values
        c_vals_i (np.ndarray): Grid of function evaluations of zeta

    Returns:
        float: _description_
    """

    # Calc normalizing constant
    norm_const = calc_grid_trapezoidal(tau_grid,
                                       c_vals_i,
                                       len(tau_grid)-1)
    
    # Get position where tau input falls on grid
    t_l = get_interval(tau_input, tau_grid)
    t_l_1 = t_l-1

    #e_i
    e_t_l = calc_grid_trapezoidal(tau_grid, c_vals_i, t_l) / norm_const
    

    e_t_l_1 = calc_grid_trapezoidal(tau_grid, c_vals_i, t_l_1) / norm_const

    e_tau_hat = (e_t_l*(tau_input - tau_grid[t_l_1]) + \
                e_t_l_1*(tau_grid[t_l]-tau_input) - \
                (tau_input-tau_grid[t_l_1])*(tau_grid[t_l]-tau_input)*(c_vals_i[t_l]-c_vals_i[t_l_1])) / \
                (tau_grid[t_l] - tau_grid[t_l_1])
    
    return e_tau_hat


@numba.njit
def logistic_transform_vector(tau_input: Union[float, np.ndarray],
                       tau_grid_expanded: np.ndarray,
                       c_vals_i: np.ndarray):
    
    # Calculat trapezium volumes
    trapz_vols = calc_grid_trapezoidal(tau_grid_expanded,
                                       c_vals_i,
                                       len(tau_grid_expanded)-1)

    # Calc normalizing constant
    norm_const = trapz_vols.sum()

    # calculate e at each tau_grid_expanded input
    e_t = trapz_vols.cumsum()

    # Now e_t omits the last point of tau_grid_expanded
    tau_grid_et = tau_grid_expanded[1:]

    # Get position where tau input falls on grid
    t_ls = get_interval_vector(tau_input, tau_grid_et).astype('int')
    t_ls_1 = t_ls-1

    # Now get the right end point and left end points for a given tau
    e_t_l = e_t[t_ls] / norm_const
    e_t_l_1 = e_t[t_ls_1] / norm_const

    # Now get interpolation tau locations
    c_t_l = get_interval_vector(tau_input, tau_grid_expanded).astype('int')
    c_t_l_1 = c_t_l-1

    # Finally calculate the logistic transform
    e_tau_hat_num = (tau_input - tau_grid_et[t_ls_1])*e_t_l + \
                    (tau_grid_et[t_ls] - tau_input)*e_t_l_1 + \
                    (c_vals_i[c_t_l] - c_vals_i[c_t_l-1])*\
                     (tau_input - tau_grid_et[t_ls_1])*(tau_grid_et[t_ls] - tau_input)

    e_tau_hat = e_tau_hat_num / (tau_grid_et[t_ls] - tau_grid_et[t_ls_1])
    
    return e_tau_hat




def base_quantile_function(tau: float,
                      mean: float,
                      sd: float,
                      v: int=1,
                      dist: str='norm') -> float:
    """ Base quantile function

    Args:
        tau (float): input quantile
        mean (float): mean of distribution
        sd (float): sd of distribution
        v (int, optional): DOF if dist='t'
        dist (str, optional): Distribution type.

    Returns:
        float: _description_
    """


    if dist=='norm':
        return norm.ppf(tau, mean, sd)
    
    elif dist == 't':
        return t.ppf(tau, df=v, loc=mean, scale=sd)
    
    else:
        print('Error')


def eta_function_i(tau_input: np.ndarray,
                 w_vals: np.ndarray,
                 tau_grid: np.ndarray,
                 mean: float,
                 sd: float,
                 v: int,
                 sigma: float,
                 dist: str='norm') -> float:
    
    
    # Calculate xi function from GP inputs w
    c_vals = np.exp(w_vals)
    
    # Apply logistic transform
    if len(c_vals.shape) > 1:  # If multiple samples of w_vals inputted (useful for plotting)
        
        # if tau_input is a vector
        if hasattr(tau_input, "__len__"): # if multiple tau values
            xi_vals = np.array([[logistic_transform(t, tau_grid, c_vals[s,:]) for t in tau_input]
                                for s in range(c_vals.shape[0])])
        else: # if single tau value
            xi_vals = np.array([logistic_transform(tau_input, tau_grid, c_vals[s,:]) for s in range(c_vals.shape[0])])
            
    else:  # if single w val (as in MCMC loop)
        if hasattr(tau_input, "__len__"): # if multiple tau values
            xi_vals = np.array([logistic_transform(t, tau_grid, c_vals) for t in tau_input])  
        else:  # if single tau value
            xi_vals = logistic_transform(tau_input, tau_grid, c_vals)
    
    # Apply base quantile function
    eta_out = sigma * base_quantile_function(xi_vals,
                                                 mean,
                                                 sd,
                                                 v=v,
                                                 dist=dist)
    
    return eta_out


def Q_joint_quantile_function(tau_input: float,
                      x_vals: np.ndarray,
                      w_samples_1: np.ndarray,
                      w_samples_2: np.ndarray,
                      sigma_1: float,
                      sigma_2: float,
                      tau_grid: np.ndarray,
                      mu: float,
                      gamma: float,
                      base_quantile_mean: float,
                      base_quantile_sd: float,
                      base_quantile_v: float=1.0,
                      base_quantile_dist: str='norm') -> np.ndarray:
    """_summary_

    Args:
        tau_input (float): _description_
        x_vals (np.ndarray): _description_
        w_samples_1 (np.ndarray): _description_
        w_samples_2 (np.ndarray): _description_
        sigma_1 (float): _description_
        sigma_2 (float): _description_
        tau_grid (np.ndarray): _description_
        mu (float): _description_
        gamma (float): _description_
        base_quantile_mean (float): _description_
        base_quantile_sd (float): _description_
        base_quantile_v (int, optional): _description_. Defaults to 1.
        base_quantile_dist (str, optional): _description_. Defaults to 'norm'.

    Returns:
        np.ndarray: _description_
    """

    
    eta_out_1 = eta_function_i(tau_input=tau_input,
                               w_vals=w_samples_1,
                               tau_grid=tau_grid,
                               mean=base_quantile_mean,
                               sd=base_quantile_sd,
                               v=base_quantile_v,
                               sigma=sigma_1,
                               dist=base_quantile_dist)
    
    eta_out_2 = eta_function_i(tau_input=tau_input,
                               w_vals=w_samples_2,
                               tau_grid=tau_grid,
                               mean=base_quantile_mean,
                               sd=base_quantile_sd,
                               v=base_quantile_v,
                               sigma=sigma_2,
                               dist=base_quantile_dist)
    
    output = mu + gamma*x_vals + \
        ((1-x_vals)/2)*eta_out_1 + \
        ((1+x_vals)/2)*eta_out_2
    
    return output

@numba.njit
def eta_function_i_vector(tau_input: np.ndarray,
                 w_vals: np.ndarray,
                 tau_grid: np.ndarray,
                 mean: float,
                 sd: float,
                 v: float,
                 sigma: float,
                 dist: str='norm'):
    
    c_vals = np.exp(w_vals)
    
    # Apply logistic transform
    xi_vals = logistic_transform_vector(tau_input, tau_grid, c_vals)

    # For xi_vals that == 1 (happens when w vals too big) remove a small constant
    if xi_vals[0] == 0:
        #print("Entered 1")
        # Find next non zero
        for j in xi_vals[1:]:
            if j > 0:
                break
        if j > 0:
            xi_vals[xi_vals==0] =j - j/2
        else:
            xi_vals[xi_vals==0] = 3.93881666e-294
            
    if xi_vals[-1] == 1:
        #print("Entered 2")
        for j in xi_vals[-2::-1]:
            if j < 1:
                break
        if j < 1:
            xi_vals[xi_vals==1] = j + (1-j)/2
        
        else:
            xi_vals[xi_vals==1] = 1-1e-16

    #xi_vals[xi_vals==1] = xi_vals[xi_vals==1] - 1e-15

    # Apply 

    if dist == 'norm':
        eta_out = sigma * numba_norm.ppf(xi_vals,
                                     mean,
                                     sd)
        
    elif dist == 't':

        eta_out = sigma * numba_t.ppf(xi_vals,
                                      float(v),
                                     mean,
                                     sd)
    
    
    return eta_out

@numba.njit
def Q_joint_quantile_function_vector(tau_input: np.ndarray,
                      x_vals: np.ndarray,
                      w_samples_1: np.ndarray,
                      w_samples_2: np.ndarray,
                      sigma_1: float,
                      sigma_2: float,
                      tau_grid: np.ndarray,
                      mu: float,
                      gamma: float,
                      base_quantile_mean: float,
                      base_quantile_sd: float,
                      base_quantile_v: float=1.0,
                      base_quantile_dist: str='norm'):
    
    
    eta_out_1 = eta_function_i_vector(tau_input=tau_input,
                               w_vals=w_samples_1,
                               tau_grid=tau_grid,
                               mean=base_quantile_mean,
                               sd=base_quantile_sd,
                               v=base_quantile_v,
                               sigma=sigma_1,
                               dist=base_quantile_dist)
    
    eta_out_2 = eta_function_i_vector(tau_input=tau_input,
                               w_vals=w_samples_2,
                               tau_grid=tau_grid,
                               mean=base_quantile_mean,
                               sd=base_quantile_sd,
                               v=base_quantile_v,
                               sigma=sigma_2,
                               dist=base_quantile_dist)
    
    output = mu + gamma*x_vals + \
        ((1-x_vals)/2)*eta_out_1 + \
        ((1+x_vals)/2)*eta_out_2
    
    return output

@numba.njit
def grid_search_deriv_approx_vector(y_i:float,
                             x_i: np.ndarray,
                             w_samples_1: np.ndarray,
                             w_samples_2: np.ndarray,
                             sigma_1: float,
                             sigma_2: float,
                             tau_grid: np.ndarray,
                             tau_grid_expanded: np.ndarray,
                             mu: float,
                             gamma: float,
                             base_quantile_mean: float,
                             base_quantile_sd: float,
                             base_quantile_v: float=1.0,
                             base_quantile_dist: str='norm'):

    
    
    Q_y_i_vals = Q_joint_quantile_function_vector(tau_input=tau_grid,
                                              x_vals=x_i,
                                              w_samples_1=w_samples_1,
                                              w_samples_2=w_samples_2,
                                              sigma_1=sigma_1,
                                              sigma_2=sigma_2,
                                              tau_grid=tau_grid_expanded,
                                              mu=mu,
                                              gamma=gamma,
                                              base_quantile_mean=base_quantile_mean,
                                              base_quantile_sd=base_quantile_sd,
                                              base_quantile_v=base_quantile_v,
                                              base_quantile_dist=base_quantile_dist)
    
    
    t_l = 0
    while True:
        if Q_y_i_vals[t_l] > y_i:
            
            break
        t_l += 1
        if t_l == len(tau_grid):
            break

    if t_l >= len(tau_grid)-1:
        tau_edge = 1
        
        Q_y_edge = Q_joint_quantile_function_vector(tau_input=np.array([tau_edge]),
                                            x_vals=x_i,
                                            w_samples_1=w_samples_1,
                                            w_samples_2=w_samples_2,
                                            sigma_1=sigma_1,
                                            sigma_2=sigma_2,
                                            tau_grid=tau_grid_expanded,
                                            mu=mu,
                                            gamma=gamma,
                                            base_quantile_mean=base_quantile_mean,
                                            base_quantile_sd=base_quantile_sd,
                                            base_quantile_v=base_quantile_v,
                                            base_quantile_dist=base_quantile_dist) 

        if y_i > Q_y_edge[0]: # If still outside boundary
            top_diff = 1e99 # Set to arbitrary big number
        
        else:
            top_diff = Q_y_edge[0] - Q_y_i_vals[-1]

        if top_diff < 0:
            print("ERRORRRR", top_diff, Q_y_edge[0], Q_y_i_vals[-1])
            ##print('w1: ', w_samples_1)
            #print('w2: ', w_samples_2)

        if top_diff == 0:
            top_diff = 1e-300
        #print("Top: ", top_diff)

        # TODO: This top diff affects alot!!!! expand grid to make value for this
        #top_diff = 0.009990000000000054
        #top_diff = 0.1
        deriv_Q_y = (top_diff)/(tau_edge - tau_grid[t_l-1])
    
    
    elif t_l == 0:
        tau_edge = 0
        
        Q_y_edge = Q_joint_quantile_function_vector(tau_input=np.array([tau_edge]),
                                    x_vals=x_i,
                                    w_samples_1=w_samples_1,
                                    w_samples_2=w_samples_2,
                                    sigma_1=sigma_1,
                                    sigma_2=sigma_2,
                                    tau_grid=tau_grid_expanded,
                                    mu=mu,
                                    gamma=gamma,
                                    base_quantile_mean=base_quantile_mean,
                                    base_quantile_sd=base_quantile_sd,
                                    base_quantile_v=base_quantile_v,
                                    base_quantile_dist=base_quantile_dist) 
        
        
        if Q_y_edge[0] < -1e99:
            print('erorr bottom no too large!!')
        if y_i < Q_y_edge[0]:  # TODO Q_edge_0 = inf??
            bot_diff = 1e300 # Set to arbitrary big number
            #print('SSS!!')
        else:
            bot_diff = Q_y_i_vals[0] - Q_y_edge[0]
        if bot_diff == 0:
            bot_diff = 1e-300
        #print('bot: ', bot_diff)

        #bot_diff = 0.05
        deriv_Q_y = (bot_diff)/(tau_grid[t_l] - tau_edge)
        #print(tau_grid[t_l])
        #if top_diff == 0:
        #    top_diff = 1e-20
    
    else:
        deriv_Q_y = (Q_y_i_vals[t_l] - Q_y_i_vals[t_l-1])/(tau_grid[t_l] - tau_grid[t_l-1])
        
        #print(Q_y_i_vals[t_l])
        #print(Q_y_i_vals[t_l-1])
        #print(t_l)
        #print(t_l-1)
        
    #print(tau_grid[t_l])
    return deriv_Q_y



#@numba.njit
def eval_ll(y_vals_true,
                 x_vals,
                 w_samples_1,
                 w_samples_2,
                 sigma_1,
                 sigma_2,
                 tau_grid,
                 tau_grid_expanded,
                 mu,
                 gamma,
                 base_quantile_mean=0.0,
                 base_quantile_sd=1.0,
                 base_quantile_v=1.0,
                 base_quantile_dist='norm',
                 splice=True,
                 b_max=None,
                 b_min=None,
                 multi_var=False,
                 proj_x=None):
    
    """ Evaluate likelihood for a vector of X and Y values
    y_vals_true: vector of n y variables
    x_vals: vector of n x variables
    splice: Whether to use splicing to evaluate out of range points
    b_max: Quantile value of tau_grid[-1]
    b_min: Quantile value of tau_grid[0]

    Returns:
        _type_: _description_
    """

    eta_1  = eta_function_i_vector(tau_input=tau_grid,
                               w_vals=w_samples_1,
                               tau_grid=tau_grid_expanded,
                               mean=base_quantile_mean,
                               sd=base_quantile_sd,
                               v=base_quantile_v,
                               sigma=sigma_1,
                               dist=base_quantile_dist)
    
    eta_2  = eta_function_i_vector(tau_input=tau_grid,
                           w_vals=w_samples_2,
                           tau_grid=tau_grid_expanded,
                           mean=base_quantile_mean,
                           sd=base_quantile_sd,
                           v=base_quantile_v,
                           sigma=sigma_2,
                           dist=base_quantile_dist)
    
    if not multi_var:
        Q_tx  = mu + gamma * np.expand_dims(x_vals, -1) + \
                    np.expand_dims((1-x_vals)/2, -1) *eta_1 + \
                    np.expand_dims((1+x_vals)/2, -1) *eta_2
    if multi_var:
        Q_tx  = mu + np.expand_dims(x_vals@gamma, -1) + \
                np.expand_dims((1-proj_x)/2, -1) *eta_1 + \
                np.expand_dims((1+proj_x)/2, -1) *eta_2

    
    # Note Q_tx: shape of n x length(tau_grid), each row corresponds to the function Q_tx_i , each column is the function evaluate at grid index
    
    # Sorted search to see where each bucket y_i falls in
    indices = np.empty(Q_tx.shape[0], dtype='int')
    for i in range(Q_tx.shape[0]):
        row = Q_tx[i]
        indices[i] = np.searchsorted(row, y_vals_true[i])
        # indices is a vector of shape n_data (Q_tx.shape[0]) each entry indicates where in the tau grid the data point i falls in
    
    
    # Check if any y_i exceeds quantile function bounds 
    if not splice:
        if np.max(indices) >= len(tau_grid): # Means there is a y_i above the max quantile
            return -np.inf  # Return arbitrarily poor loglikelihood

        # Check if any y_i exceeds quantile function bounds 
        if np.min(indices) == 0: # Means there is a y_i b elow the min quantile
        
            return -np.inf  # Return arbitrarily poor loglikelihood
        

        # If not all is good
        top_approx=np.zeros(len(indices))
        bot_approx=np.zeros(len(indices))

        for i in range(len(indices)):
            top_approx[i] = Q_tx[i,indices[i]]
            bot_approx[i] = Q_tx[i,indices[i]-1]
        
        # Lastly get tau length
        interval_lengths = tau_grid[indices] - tau_grid[indices-1]

        # These represent (unlogged values and negative)
        deriv_Q_y = (top_approx-bot_approx)/interval_lengths

        # Sometimes if if sampled values too big eta function becomes flat and derivative is zero then we set a 
        # very small number else log function generates nans
        deriv_Q_y[deriv_Q_y==0] = 1e-300
        
        ll = (-1*np.log(deriv_Q_y)).sum()
        
        return ll

    else:
        # Indices contains which bucket in Q_tx the y value falls in

        # split data into 3 groups, over, under and in range
        over_points = np.where(indices==len(tau_grid))[0]
        under_points = np.where(indices==0)[0]
        in_range_points = np.where((indices>0) & (indices < len(tau_grid)))[0]

        # Spliced distribution likelihood for points that are over
        if len(over_points) > 0:
            s = 0.1
            Q_end_vals = Q_tx[:,-1]  # Get the values of Q_tx at the largest tau
            a_max = Q_end_vals[over_points]  # Get the values only for points in over_points 
            y_over = y_vals_true[over_points]
            b_max = numba_t.ppf(np.array([tau_grid[-1]]), loc=0.0,
                                df=1.0,
                                scale=1.0)

            ll_over = numba_t.logpdf(b_max + (y_over-a_max)/s, df=1.0,loc=0.0,scale=1.0) - np.log(s)
            #if np.any(np.isnan(ll_over)):
            #    print('error ll-over nan eval_ll')
            #    print('b:', b_max)
            #    print('y: ', y_over)
            #    print("Qtx: ", Q_tx)
            #    print("Proj x: ", proj_x)
            #    print("Gamma: ", gamma)
            #    print('a: ', a_max)
            #    print('s: ', s)
        else:
            ll_over = np.array([0])


        # Spliced distribution likelihood of points that are under
        if len(under_points) > 0:
            s = 0.1
            Q_start_vals = Q_tx[:,0]  # Get the values of Q_tx at the smallest tau
            a_min = Q_start_vals[under_points]  # Get the values only for points in over_points 
            y_under = y_vals_true[under_points]
            b_min = numba_t.ppf(np.array([[tau_grid[0]]]), loc=0.0,df=1.0,scale=1.0)

            ll_under = numba_t.logpdf(b_min + (y_under-a_min)/s, df=1.0,loc=0.0,scale=1.0) - np.log(s)
            #if np.any(np.isnan(ll_under)):
            #    print('error ll-under nan eval_ll')
            #    print('b:', b_min)
            #    print('y: ', y_under)
            #    print("Qtx: ", Q_tx)
            #    print("Proj x: ", proj_x)
            #    print("Gamma: ", gamma)
            #    print('a: ', a_min)
            #    print('s: ', s)
        else:
            ll_under = np.array([0])
        
        # If not all is good
        top_approx=np.zeros(len(in_range_points))
        bot_approx=np.zeros(len(in_range_points))

        for i in range(len(in_range_points)):
            point_id = in_range_points[i]
            top_approx[i] = Q_tx[point_id,indices[point_id]]
            bot_approx[i] = Q_tx[point_id,indices[point_id]-1]

            if bot_approx[i] > top_approx[i]:
                print("Upper Out: ") 
                print(*over_points, sep=',')
                print("Lower Out: ")
                print(*under_points, sep=',')
                print("In Range: ")
                print(*in_range_points, sep=',')
                print("Q Info")
                print(bot_approx[i], top_approx[i])
                print('eta_1: ')
                print(*eta_1, sep=',')
                print('eta_2: ')
                print(*eta_2, sep=',')
                print('x_vals: ')
                print(*x_vals,sep=',')
                print(mu,gamma)
                print(Q_tx.shape)
                print("Q_tx :", )
                print(*Q_tx[i],sep=',')
                print("Data id: ", i)
                print(indices[point_id])
        
        # Lastly get tau length
        interval_lengths = tau_grid[indices[in_range_points]] - tau_grid[indices[in_range_points]-1]

        # These represent (unlogged values and negative)
        deriv_Q_y = (top_approx-bot_approx)/interval_lengths

        # Sometimes if if sampled values too big eta function becomes flat and derivative is zero then we set a 
        # very small number else log function generates nans
        deriv_Q_y[deriv_Q_y==0] = 1e-300
        
        ll = (-1*np.log(deriv_Q_y)).sum() + ll_over.sum() + ll_under.sum()
        
        if np.isnan(ll):
            print("LL Under: ", ll_under)
            print("LL Over: ", ll_over)
            print("Deriv Q: ", (deriv_Q_y))

        return ll


def eval_ll_eta(y_vals_true,
                 x_vals,
                 eta_1,
                 eta_2,
                 sigma_1,
                 sigma_2,
                 tau_grid,
                 tau_grid_expanded,
                 mu,
                 gamma,
                 base_quantile_mean=0.0,
                 base_quantile_sd=1.0,
                 base_quantile_v=1.0,
                 base_quantile_dist='norm',
                 splice=True,
                  b_max=None,
                 b_min=None,
                 multivariate_x=False,
                 proj_x=None):

    
    if multivariate_x:
        Q_tx  = mu + np.expand_dims(x_vals@gamma, -1) + \
                np.expand_dims((1-proj_x)/2, -1) *eta_1 + \
                np.expand_dims((1+proj_x)/2, -1) *eta_2
    else:
        Q_tx  = mu + gamma * np.expand_dims(x_vals, -1) + \
                np.expand_dims((1-x_vals)/2, -1) *eta_1 + \
                np.expand_dims((1+x_vals)/2, -1) *eta_2
        
    # Sorted search to see where each bucket y_i falls in
    indices = np.empty(Q_tx.shape[0], dtype='int')
    for i in range(Q_tx.shape[0]):
        row = Q_tx[i]
        indices[i] = np.searchsorted(row, y_vals_true[i])
    
    # Check if any y_i exceeds quantile function bounds 
    if not splice:
        if np.max(indices) >= len(tau_grid): # Means there is a y_i above the max quantile
            return -np.inf  # Return arbitrarily poor loglikelihood

        # Check if any y_i exceeds quantile function bounds 
        if np.min(indices) == 0: # Means there is a y_i below the min quantile
        
            return -np.inf  # Return arbitrarily poor loglikelihood
        

        # If not all is good
        top_approx=np.zeros(len(indices))
        bot_approx=np.zeros(len(indices))

        for i in range(len(indices)):
            top_approx[i] = Q_tx[i,indices[i]]
            bot_approx[i] = Q_tx[i,indices[i]-1]
        
        # Lastly get tau length
        interval_lengths = tau_grid[indices] - tau_grid[indices-1]

        # These represent (unlogged values and negative)
        deriv_Q_y = (top_approx-bot_approx)/interval_lengths

        # Sometimes if if sampled values too big eta function becomes flat and derivative is zero then we set a 
        # very small number else log function generates nans
        deriv_Q_y[deriv_Q_y==0] = 1e-300
        
        ll = (-1*np.log(deriv_Q_y)).sum()
        
        return ll

    else:
        # Indices contains which bucket in Q_tx the y value falls in
        # split data into 3 groups, over, under and in range
        over_points = np.where(indices==len(tau_grid))[0]
        under_points = np.where(indices==0)[0]
        in_range_points = np.where((indices>0) & (indices < len(tau_grid)))[0]

        # Spliced distribution likelihood for points that are over
        if len(over_points) > 0:
            s = 0.1
            Q_end_vals = Q_tx[:,-1]  # Get the values of Q_tx at the largest tau
            a_max = Q_end_vals[over_points]  # Get the values only for points in over_points 
            y_over = y_vals_true[over_points]
            b_max = numba_t.ppf(np.array([tau_grid[-1]]), loc=0.0,
                                df=1.0,
                                scale=1.0)

            ll_over = numba_t.logpdf(b_max + (y_over-a_max)/s, df=1.0,loc=0.0,scale=1.0) - np.log(s)
            #if np.isnan(ll_over):
            #    print('error ll-over nan eval_ll_eta')
            #    print('b:', b_max)
            #    print('y: ', y_over)
            #    print("Qtx: ", Q_tx)
            #    print("Proj x: ", proj_x)
            #    print("Gamma: ", gamma)
            #    print('a: ', a_max)
            #    print('s: ', s)
        else:
            ll_over = np.array([0])


        # Spliced distribution likelihood of points that are under
        if len(under_points) > 0:
            s = 0.1
            Q_start_vals = Q_tx[:,0]  # Get the values of Q_tx at the smallest tau
            a_min = Q_start_vals[under_points]  # Get the values only for points in over_points 
            y_under = y_vals_true[under_points]
            b_min = numba_t.ppf(np.array([[tau_grid[0]]]), loc=0.0,df=1.0,scale=1.0)

            ll_under = numba_t.logpdf(b_min + (y_under-a_min)/s, df=1.0,loc=0.0,scale=1.0) - np.log(s)
            #if np.isnan(ll_under):
            #    print('error ll-under nan eval_ll_eta')
            #    print('b:', b_min)
            #    print('y: ', y_under)
            #    print("Qtx: ", Q_tx)
            #    print("Proj x: ", proj_x)
            #    print("Gamma: ", gamma)
            #    print('a: ', a_min)
            #    print('s: ', s)
        else:
            ll_under = np.array([0])
        
        # If not all is good
        top_approx=np.zeros(len(in_range_points))
        bot_approx=np.zeros(len(in_range_points))

        for i in range(len(in_range_points)):
            point_id = in_range_points[i]
            top_approx[i] = Q_tx[point_id,indices[point_id]]
            bot_approx[i] = Q_tx[point_id,indices[point_id]-1]
        
        # Lastly get tau length
        interval_lengths = tau_grid[indices[in_range_points]] - tau_grid[indices[in_range_points]-1]

        # These represent (unlogged values and negative)
        deriv_Q_y = (top_approx-bot_approx)/interval_lengths

        # Sometimes if if sampled values too big eta function becomes flat and derivative is zero then we set a 
        # very small number else log function generates nans
        deriv_Q_y[deriv_Q_y==0] = 1e-300
        
        ll = (-1*np.log(deriv_Q_y)).sum() + ll_over.sum() + ll_under.sum()
        
        return ll
    

def generate_beta_samples(tau_input: float,
                          tau_grid: np.ndarray,
                          w_approx_store: List[np.ndarray],
                          mu_store: List[float],
                          gamma_store: List[float],
                          sigma_1_store: List[float],
                          sigma_2_store: List[float],
                          base_quantile_dist:str = 't',
                          v_store: List[float] = None):

    beta_0_store = []
    beta_1_store = []
    for i in range(0,len(w_approx_store)):
        w_samp = w_approx_store[i]
        L = int(w_samp.shape[0]/2)
        w1_samp = w_samp[0:L]
        w2_samp = w_samp[L:]

        mu_samp = mu_store[i]
        gamma_samp = gamma_store[i]
        sigma_1_samp = sigma_1_store[i]
        sigma_2_samp = sigma_2_store[i]

        if v_store:
            v_samp = v_store[i]
        else:
            v_samp = 1.0

        eta_1_samp = eta_function_i_vector(tau_input=np.array([tau_input]),
                                             w_vals=w1_samp,
                                             tau_grid=tau_grid,
                                             mean=0.0,
                                             sd=1.0,
                                             v=v_samp,
                                             sigma=sigma_1_samp,
                                             dist=base_quantile_dist)[0]


        eta_2_samp = eta_function_i_vector(tau_input=np.array([tau_input]),
                                             w_vals=w2_samp,
                                             tau_grid=tau_grid,
                                             mean=0.0,
                                             sd=1.0,
                                             v=v_samp,
                                             sigma=sigma_2_samp,
                                             dist=base_quantile_dist)[0]


        beta_0_samp = mu_samp + (eta_1_samp + eta_2_samp)/2
        beta_1_samp = gamma_samp + (eta_2_samp - eta_1_samp)/2

        beta_0_store.append(beta_0_samp)
        beta_1_store.append(beta_1_samp)
        
    return beta_0_store, beta_1_store
    

def logpdf_mvn(x, mean, cov):
    """
    Compute the loglikelihood of a multivariate normal distribution (MND).
    
    Parameters:
    x (numpy array): A 1-D numpy array of data points.
    mean (numpy array): A 1-D numpy array representing the mean vector of the MND.
    cov (numpy array): A 2-D numpy array representing the covariance matrix of the MND.
    
    Returns:
    float: The loglikelihood of the MND.
    """
    n = x.shape[0]
    diff = x - mean
    return -0.5 * (n * np.log(2 * np.pi) + np.log(np.linalg.det(cov)) + diff.T @ np.linalg.inv(cov) @ diff)


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


def geometric_seq(start, end, ratio, upper=True):
    # Generates geometric sequence for tau grid formation
    output_vals = []
    current_val = start
    i = 0
    
    if upper:
        while current_val < end: 
            current_val += ratio/(2**i)
            i += 1
            output_vals.append(current_val)
    
    else:
        while current_val > end:
            current_val -= ratio/(2**i)
            i += 1
            output_vals.append(current_val)
        
        
    return np.array(output_vals)



def phi_prior_sample(cov_mat, alpha_kappa, beta_kappa,
                      prior_on_t, multivariate_x, n_covar=None):
    
    # Samples a value for w1_knot, w2_knot, mu, gamma, sigma1,sigma2
    # w1_knot,w2_knot are from t distribution
    m = cov_mat.shape[0]
    w1_knot_sample = multivariate_t.rvs(loc=np.zeros(m),
                      shape=cov_mat*(beta_kappa/alpha_kappa),
                      df=2*alpha_kappa)
    
    w2_knot_sample = multivariate_t.rvs(loc=np.zeros(m),
                      shape=cov_mat*(beta_kappa/alpha_kappa),
                      df=2*alpha_kappa)
    
    
    # mu, gamma from t(0,1), sigma1,sigma2 are from gamma()
    mu_samp = student_t_univar.rvs(1,0,1)
    sigma1_samp = np.sqrt(gamma.rvs(a=2,scale=1/2))
    sigma2_samp = np.sqrt(gamma.rvs(a=2,scale=1/2))

    if multivariate_x:
        # sample gamma multvariate and x_alpha
        gamma_samp = np.array([student_t_univar.rvs(1,0,1) for _ in range(n_covar)])
        x_alpha_samp = multivariate_t.rvs(loc = np.zeros(n_covar),
                                            shape=np.eye(n_covar),
                                            df=1)
        if prior_on_t:
            v_samp = gamma.rvs(a=3,scale=1/2) 

            prior_phi_samp = np.concatenate([w1_knot_sample,
                                            w2_knot_sample,
                                            np.array([mu_samp]),
                                            gamma_samp,
                                            np.array([sigma1_samp,
                                            sigma2_samp,
                                            v_samp]),
                                            x_alpha_samp])
            
        else:
            prior_phi_samp = np.concatenate([w1_knot_sample,
                                        w2_knot_sample,
                                        np.array([mu_samp]),
                                        gamma_samp,
                                        np.array([sigma1_samp,
                                        sigma2_samp])],
                                        x_alpha_samp)
    else:
        # sample gamma unviariate
        gamma_samp = student_t_univar.rvs(1,0,1)

        if prior_on_t:
            v_samp = gamma.rvs(a=3,scale=1/2) 

            prior_phi_samp = np.concatenate([w1_knot_sample,
                                            w2_knot_sample,
                                            np.array([mu_samp,
                                            gamma_samp,
                                            sigma1_samp,
                                            sigma2_samp,
                                            v_samp])])
            
        else:
            prior_phi_samp = np.concatenate([w1_knot_sample,
                                        w2_knot_sample,
                                        np.array([mu_samp,
                                        gamma_samp,
                                        sigma1_samp,
                                        sigma2_samp])])
    return prior_phi_samp



def geometric_seq(start, end, ratio, upper=True):
    # Generates geometric sequence for tau grid formation
    output_vals = []
    current_val = start
    i = 0
    
    if upper:
        while current_val < end: 
            current_val += ratio/(2**i)
            i += 1
            output_vals.append(current_val)
    
    else:
        while current_val > end:
            current_val -= ratio/(2**i)
            i += 1
            output_vals.append(current_val)
        
    return np.array(output_vals)


def project_x(X_covar, X_alpha, return_ab = False):
    """
    X_covar: n x p covariate matrix (not including intercept!)
    X_lpha: p dim projection vector (randomly sampled)
    """
    
    if X_covar.shape == 1:
        print("Error")
    if X_covar.shape[1]==1:
        print("Error")
    # Initial projection to calculate a and b
    init_proj = (X_covar @ X_alpha)
    
    if len(X_covar) == 1:  # point will just evaluate to 0 otherwise
        # check with surya!!!!
        return init_proj
    
    # Calc a
    X_a = (np.max(init_proj) + np.min(init_proj))/2
    # Calc b
    X_b = (np.max(init_proj) - np.min(init_proj))/2
    
    
    # Calc final projection
    X_proj = (init_proj - X_a)/X_b
    
    if return_ab:
        return X_proj, X_a, X_b
    else:
        return X_proj


def generate_beta_samples_multivar(tau_input: float,
                                      tau_grid: np.ndarray,
                                      w_approx_store: List[np.ndarray],
                                      mu_store: List[float],
                                      gamma_store: List[np.ndarray],
                                      sigma_1_store: List[float],
                                      sigma_2_store: List[float],
                                      X_alpha_store: List[np.ndarray],
                                      base_quantile_dist:str = 't',
                                      v_store: List[float] = None,
                                    X_vals=None):

    beta_1s_store = []
    for i in range(0,len(w_approx_store)):
        w_samp = w_approx_store[i]
        L = int(w_samp.shape[0]/2)
        w1_samp = w_samp[0:L]
        w2_samp = w_samp[L:]

        mu_samp = mu_store[i]
        gamma_samp = gamma_store[i]
        sigma_1_samp = sigma_1_store[i]
        sigma_2_samp = sigma_2_store[i]
        X_alpha_samp = X_alpha_store[i]
        if v_store:
            v_samp = v_store[i]
        else:
            v_samp = 1.0

        eta_1_samp = eta_function_i_vector(tau_input=np.array([tau_input]),
                                             w_vals=w1_samp,
                                             tau_grid=tau_grid,
                                             mean=0.0,
                                             sd=1.0,
                                             v=v_samp,
                                             sigma=sigma_1_samp,
                                             dist=base_quantile_dist)[0]


        eta_2_samp = eta_function_i_vector(tau_input=np.array([tau_input]),
                                             w_vals=w2_samp,
                                             tau_grid=tau_grid,
                                             mean=0.0,
                                             sd=1.0,
                                             v=v_samp,
                                             sigma=sigma_2_samp,
                                             dist=base_quantile_dist)[0]
        
        # Get b in projection
        init_proj = (X_vals @ X_alpha_samp)
        # Calc a
        X_a = (max(init_proj) + min(init_proj))/2
        # Calc b
        X_b = (max(init_proj) - min(init_proj))/2

        beta_1_samp = gamma_samp + ((eta_2_samp - eta_1_samp)/(2*X_b)) * X_alpha_samp

        beta_1s_store.append(beta_1_samp)
        
    return beta_1s_store