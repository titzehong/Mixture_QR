# Functions to process a set of MCMC samples 
import numpy as np
from typing import List
from sklearn.cluster import AgglomerativeClustering
import pandas as pd


def calc_hit_matrix_sample(partition:np .array) -> np.array:
    """ calculates a co-occurence matrix for a given sample 
    partition: np.array (1d) of cluster indices of data

    Args:
        partition (np.array): An array (n dim) of hard clustering indices for a set of n data points

    Returns:
        np.array: n x n binary indicator matrix, with 1 when points i,j are in same cluster
    """    
    
    # Form hit matrix
    n_part = len(partition)
    hit_matrix = np.zeros([n_part,n_part])
    for ii in range(n_part):
        for jj in range(n_part):
            hit_matrix[ii,jj] = (partition[ii]==partition[jj])
    return hit_matrix

def calc_hit_matrix(partition_samples:List[np.array],
                    burn_samples:int=0,
                    normalize:bool=True) -> np.array:
    """ From a set of MCMC samples of hard clusterings compute counts of co-occurence (normalize = False) or
    or normalize counts of co-occurence(normalize = True)

    Args:
        partition_samples (List[np.array]): List contatining samples of hard clsuterings
        burn_samples (int, optional): number of samples to discard at the start. Defaults to 0.
        normalize (bool, optional): Whether to normalize. Defaults to True.

    Returns:
        np.array: Counts or normalized counts.
    """    
    # for a given set of partition samples, calculate the number of times 
    n_partitions = len(partition_samples)

    n = len(partition_samples[0])  # No. of data points in each sample
    hit_matrix_overall = np.zeros([n,n])
    n_samples_used = n_partitions - burn_samples

    for i in range(burn_samples,n_partitions):
        hit_matrix_overall += calc_hit_matrix_sample(partition_samples[i])
    
    if normalize:
        return (1/n_samples_used)*hit_matrix_overall
    
    else:
        return hit_matrix_overall


def agglo_cluster(sim_matrix: np.array,n_clust:int,
                  linkage_type:str='average') -> np.array:

    """ Applys hierarchical clustering to a similarity matrix (sim_matrix),
        generating n_clust numbers of clusters

    Args:
        sim_matrix (np.array): similarity matrix, (0-1) range for each element. Eg output from calc_hit_matrix with normalize=True
        n_clust (int): number of clusters wanted
        linkage_type (str, optional): Type of linkage to use, average seems to work best. Defaults to 'average'.

    Returns:
        np.array: _description_
    """    
    model = AgglomerativeClustering(
        affinity='precomputed',
        n_clusters=n_clust,
        linkage=linkage_type).fit(1-sim_matrix)
    
    return model.labels_



def calc_c_vals(cluster_sol, similarity_mat, Ys):
    
    c_vals = np.zeros(len(Ys))

    for i in range(len(Ys)):

        point_clust = cluster_sol[i]
        point_sims = similarity_mat[i]

        neighbs = np.where(cluster_sol==point_clust)[0]
        neighbs_sim = point_sims[neighbs]
        c_val_i = np.mean(neighbs_sim)
        c_vals[i] = c_val_i
        
    return c_vals
    

def confidence_matrix(cluster_sol, similarity_mat, Ys, c_vals):
    
    # Get names of clusters
    sorted_clust_names = pd.Series(cluster_sol).value_counts().index

    # Re-arrange IDs and form new sorted similarity matrix
    arranged_ids = []
    for c in sorted_clust_names:

        clust_ids = np.where(cluster_sol==c)[0]
        c_vals_clust = c_vals[clust_ids]
        sorted_clust_ids =clust_ids[np.argsort(c_vals_clust)[::-1]]

        arranged_ids.append(sorted_clust_ids)

    arranged_ids = np.concatenate(arranged_ids)

    sorted_matrix = np.zeros((len(Ys),len(Ys)))

    for i in range(len(Ys)):
        for j in range(len(Ys)):

            sorted_matrix[i,j] = similarity_mat[arranged_ids[i], arranged_ids[j]]
    

    return sorted_matrix