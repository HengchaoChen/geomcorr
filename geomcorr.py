# Quotient geometry for correlation matrix and related functions
# Author: Hengchao Chen
# 
# Updated: 2024-07-06

"""
This module is used to calculate the quotient geometry for correlation matrix.

The module contains the following functions:

    - distance: Compute the distance between two correlation matrices.
    - pairwise_distance: Compute the pairwise distance between two sets of correlation matrices.
    - frechet_mean: Compute the Frechet mean of a set of correlation matrices.
    - frechet_variance: Compute the Frechet variance of a set of correlation matrices.
    - clustering: Perform clustering on a set of correlation matrices.
    
The module also addresses these questions:

    - How to implement Riemannian optimization efficiently? Choose a suitable initialization."""

import numpy as np
from scipy.linalg import sqrtm

import sklearn.metrics
from scipy.cluster.hierarchy import linkage, fcluster

# ------------------- Dimension Checking -------------------

def check_dimension(X):
    """Check the dimension of the input matrix.
    
    Parameters
    ----------
    X : 2D array (m * m correlation matrix) representing a correlation matrix. Or 3D array (n_samples, m * m correlation matrices) representing correlation matrices.
    
    Returns
    -------
    X : 3D array (n_samples, m * m correlation matrices) representing correlation matrices."""

    if X.ndim == 2:
        if X.shape[0] != X.shape[1]:
            raise ValueError("The input matrix must be square.")
        else:
            return X.reshape(1, X.shape[0], X.shape[1])

    if X.ndim != 3:
        raise ValueError("The input matrix must be a 3D array.")
    
    if X.shape[1] != X.shape[2]:
        raise ValueError("The input matrix must be square.")
    
    return X

def check_dim_twosets(X, Y):
    """Compare two sets of correlation matrices.
    
    Parameters
    ----------
    X : 3D array (n_samples, m * m correlation matrices) representing correlation matrices.
    Y : 3D array (n_samples, m * m correlation matrices) representing correlation matrices.
    
    Returns
    -------
    X : 3D array (n_samples, m * m correlation matrices) representing correlation matrices.
    Y : 3D array (n_samples, m * m correlation matrices) representing correlation matrices."""

    X = check_dimension(X)
    Y = check_dimension(Y)

    if X.shape != Y.shape:
        raise ValueError("The two matrices must have the same shape.")
    else:
        return X, Y
    
def check_pairwise_dim(X, Y):
    """Compare two sets of correlation matrices for pairwise distance.
    
    Parameters
    ----------
    X : 3D array (n_X, m * m correlation matrices) representing correlation matrices.
    Y : 3D array (n_Y, m * m correlation matrices) representing correlation matrices.
    
    Returns
    -------
    X : 3D array (n_X, m * m correlation matrices) representing correlation matrices.
    Y : 3D array (n_Y, m * m correlation matrices) representing correlation matrices."""

    X = check_dimension(X)
    Y = check_dimension(Y)

    if X.shape[1] != Y.shape[1] or X.shape[2] != Y.shape[2]:
        raise ValueError("The two matrices must have the same shape.")
    else:
        return X, Y
    
def check_base_dim(X, base):
    """Compare a list of correlation matrices with a base correlation matrix.
    
    Parameters
    ----------
    X : 3D array (n_samples, m * m correlation matrices) representing correlation matrices.
    base : 2D array (m * m correlation matrix) representing the base correlation matrix. Or 3D array (1, m * m correlation matrices) representing the base correlation matrix.
    
    Returns
    -------
    X : 3D array (n_samples, m * m correlation matrices) representing correlation matrices.
    base : 3D array (n_samples, m * m correlation matrices) representing the base correlation matrix."""

    X = check_dimension(X)
    base = check_dimension(base)

    if base.shape[0] != 1:
        raise ValueError("The base matrix must be a single correlation matrix.")
    
    base = np.repeat(base, X.shape[0], axis = 0)
    
    return X, base

# ------------------- Linear Algebra Tools -------------------

def sort_eigh_descending(matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    
    # Reverse the order
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    
    return eigenvalues, eigenvectors

def factorization(X, rank, auto_normalize = True):
    """Compute the factorization of a correlation matrix.
    
    Parameters
    ----------
    X : 2D array (m * m correlation matrix) representing a correlation matrix.
    rank : int representing the parameter of the quotient geometry
    
    Returns
    -------
    X_factor: 2D array (m * rank matrix) such that X_factor @ X_factor.T = X."""

    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError("The input matrix must be a 2D array.")
    
    if rank > X.shape[0]:
        raise ValueError("The rank must be less than the dimension of the input matrix.")
    
    eigenvalues, eigenvectors = sort_eigh_descending(X)

    X_factor = eigenvectors[:, :rank] @ np.diag(np.sqrt(np.clip(eigenvalues[:rank], 0 , None)))

    # The factorization is automatically normalized so that the rows have unit norm

    if auto_normalize:

        X_factor /= np.linalg.norm(X_factor, axis = 1, keepdims = True)

    return X_factor

def low_rank_approximation(X, rank):
    """Compute the low-rank approximation of a covariance matrix.
    
    Parameters
    ----------
    X : 2D array (m * m covariance matrix) representing a covariance matrix.
    rank : int representing the rank of the approximation.
    
    Returns
    -------
    X_approx : 2D array (m * m covariance matrix) representing the low-rank approximation of the covariance matrix."""

    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError("The input matrix must be a 2D array.")
    
    if rank > X.shape[0]:
        raise ValueError("The rank must be less than the dimension of the input matrix.")
    
    eigenvalues, eigenvectors = sort_eigh_descending(X)

    X_approx = eigenvectors[:, :rank] @ np.diag(eigenvalues[:rank]) @ eigenvectors[:, :rank].T

    return X_approx 

def covariance_to_correlation(cov_matrix):
    """Convert a covariance matrix to a correlation matrix."""
    # Compute the standard deviations
    std_dev = np.sqrt(np.diag(cov_matrix))
    
    # Outer product of standard deviations
    std_dev_outer = np.outer(std_dev, std_dev)
    
    # Element-wise division of the covariance matrix by the outer product of standard deviations
    corr_matrix = cov_matrix / std_dev_outer
    
    # Replace NaNs with 0 (in case of division by zero)
    corr_matrix[np.isnan(corr_matrix)] = 0
    
    return corr_matrix

# ------------------- Optimization Tools -------------------

def random_orthogonal_matrix(n):
    """
    Generate a random n x n orthogonal matrix.
    """
    # Generate a random n x n matrix
    A = np.random.randn(n, n)
    
    # Compute the QR decomposition
    Q, R = np.linalg.qr(A)
    
    # Q is orthogonal, but we need to ensure it has determinant 1
    # Multiply the first column by sign(det(Q))
    Q[:, 0] *= np.sign(np.linalg.det(Q))
    
    return Q

def compute_gradient(X, Y, O):
    """Compute the gradient of the distance between two factorizations in the quotient space.
    
    Parameters
    ----------
    X : 2D array (m * rank) representing the factorization of a correlation matrix.
    Y : 2D array (m * rank) representing the factorization of a correlation matrix.
    O : 2D array (rank * rank) representing an orthogonal matrix.
    
    Returns
    -------
    grad : 2D array (rank * rank) representing the gradient of the distance between the factorizations."""

    Z = X @ O

    A = np.zeros((O.shape[0], O.shape[1]))

    inner_product = np.clip(np.diag(Z @ Y.T), -1, 0.999999) # m dimensional vector

    dist = np.arccos(inner_product) 

    for i in range(X.shape[0]):

        A -= dist[i] / np.sqrt(1 - inner_product[i] ** 2) * np.outer(X[i,:], Y[i,:]) # k by k matrix

    grad = O @ (O.T @ A - A.T @ O) / 2

    return grad / X.shape[0]

def retract(O, grad, alpha):
    """Perform the retraction operation in the space of orthogonal matrices.
    
    Parameters
    ----------
    O : 2D array (rank * rank) representing an orthogonal matrix.
    grad : 2D array (rank * rank) representing the gradient.
    alpha : real number representing the step size.
    
    Returns
    -------
    O_new : 2D array (rank * rank) representing the new orthogonal matrix."""

    O_new = O + alpha * grad

    O_new, _ = np.linalg.qr(O_new)

    return O_new

def dist_prod_sphere(X, Y):
    """Compute the distance between two factorizations in the product sphere PS.
    
    Parameters
    ----------
    X : 2D array (m * rank) representing the factorization of a correlation matrix.
    Y : 2D array (m * rank) representing the factorization of a correlation matrix.
    
    Returns
    -------
    dist : real number representing the distance between the factorizations."""

    inner_product = np.clip(np.diag(X @ Y.T), -1, 1) # m dimensional vector

    dist = np.arccos(inner_product) 

    return np.linalg.norm(dist)

def dist_quotient(X, Y, T = 30, num_trial = 0, report_error = False, report_rotation = False):
    """Compute the distance between two factorizations in the quotient space.

    Parameters
    ----------
    X : 2D array (m * rank) representing the factorization of a correlation matrix.
    Y : 2D array (m * rank) representing the factorization of a correlation matrix.
    T : int representing the number of iterations.
    num_trial : int representing the number of trials. By default, num_trial = 0.

    Returns
    -------
    dist : real number representing the distance between the factorizations.
    O_min : 2D array (rank * rank) representing the orthogonal matrix minimizing dist(XO, Y).
    error_all : 1D array (num_trial + 1) representing the error of each trial."""

    alpha_init = 0.02

    # trial 1 - use a suitable rotation matrix as the initial guess

    U, _, VT = np.linalg.svd(X.T @ Y)

    O = U @ VT

    error_bar = dist_prod_sphere(X @ O, Y)

    alpha = np.copy(alpha_init) 

    for _ in range(T):

        grad = compute_gradient(X, Y, O) 

        O_new = retract(O, grad, alpha)

        error_new = dist_prod_sphere(X @ O_new, Y)

        if error_new < error_bar:

            O = np.copy(O_new)

            error_bar = np.copy(error_new)

        elif alpha > 0.005:

            alpha /= 2

            alpha_init = np.copy(alpha)

        else:

            break

    O_min = np.copy(O)

    error_min = error_bar

    error_all = np.zeros(num_trial + 1) 

    error_all[num_trial] = error_bar

    if num_trial > 0: # by default, num_trial = 0

        # trial 2 - use a random orthogonal matrix as the initial guess
        
        for i in range(num_trial):

            O = random_orthogonal_matrix(X.shape[1])

            alpha = 0.01

            error_bar = dist_prod_sphere(X @ O, Y)

            for _ in range(T):

                grad = compute_gradient(X, Y, O)

                O_new = retract(O, grad, alpha)

                error_new = dist_prod_sphere(X @ O_new, Y)

                if error_new < error_bar:

                    O = np.copy(O_new)

                    error_bar = np.copy(error_new)

                elif alpha > 0.0005:

                    alpha /= 2
                
                else:

                    break

            error_all[i] = error_bar

            if error_bar < error_min:

                O_min = np.copy(O)

                error_min = error_bar
    
    if error_min != dist_prod_sphere(X @ O_min, Y):
        raise ValueError("The code is wrong.")

    if report_error:

        if report_rotation:

            return error_min, error_all, O_min
    
        else:

            return error_min, error_all

    else:

        if report_rotation:

            return error_min, O_min

        else:

            return error_min

# ------------------- Quotient Geometry for Correlation Matrix -------------------


def distance(X, Y, rank):
    """Compute the distance between two correlation matrices.

    Parameters
    ----------
    X : 3D array (n_samples, m * m correlation matrices) representing correlation matrices.
    Y : 3D array (n_samples, m * m correlation matrices) representing correlation matrices.
    rank : int representing the parameter of the quotient geometry

    Returns
    -------
    dist : 1D array (n_samples) representing the distance between the correlation matrices."""

    X, Y = check_dim_twosets(X, Y)
    
    n_samples = X.shape[0]

    dist = np.zeros(n_samples)

    for i in range(n_samples):
        X_factor = factorization(X[i,...], rank)
        Y_factor = factorization(Y[i,...], rank)
        
        if np.linalg.norm(X_factor - Y_factor) < 1e-10:
            dist[i] = 0
        else:
            dist[i] = dist_quotient(X_factor, Y_factor)

    return dist

def pairwise_distance(X, Y, rank):

    """Compute the pairwise distance between two sets of correlation matrices.
    
    Parameters
    ----------
    X : 3D array (n_X, m * m correlation matrices) representing correlation matrices.
    Y : 3D array (n_Y, m * m correlation matrices) representing correlation matrices.
    rank : int representing the parameter of the quotient geometry
    
    Returns
    -------
    dist : 2D array (n_X, n_Y) representing the pairwise distance between the correlation matrices."""

    X = check_dimension(X)
    Y = check_dimension(Y)

    n_X = X.shape[0]
    n_Y = Y.shape[0]

    dist = np.zeros((n_X, n_Y))

    for i in range(n_X):
        for j in range(n_Y):
            X_factor = factorization(X[i,...], rank)
            Y_factor = factorization(Y[j,...], rank)

            if np.linalg.norm(X_factor - Y_factor) < 1e-10:
                dist[i, j] = 0
            else:
                dist[i, j] = dist_quotient(X_factor, Y_factor)

    return dist

def frechet_mean(X, rank, iter = 10):
    """Compute the Frechet mean of a set of correlation matrices.
    
    Parameters
    ----------
    X : 3D array (n_samples, m * m correlation matrices) representing correlation matrices.
    rank : int representing the parameter of the quotient geometry
    
    Returns
    -------
    X_mean : 2D array (m * m correlation matrix) representing the Frechet mean of the correlation matrices."""

    X = check_dimension(X)

    X_factor = np.zeros((X.shape[0], X.shape[1], rank))

    for i in range(X.shape[0]):

        X_factor[i,...] = factorization(X[i,...], rank)

    # initialization

    X_FM_factor = X_factor[0,...]

    # alternating minimization

    for i in range(iter):

        # align each factor to the Frechet mean factor

        for j in range(X.shape[0]):

            _, O_j = dist_quotient(X_factor[j,...], X_FM_factor, report_rotation = True)

            X_factor[j,...] = X_factor[j,...] @ O_j
        
        # update the Frechet mean factor

        X_FM_factor = np.mean(X_factor, axis = 0)

        X_FM_factor /= np.linalg.norm(X_FM_factor, axis = 1, keepdims = True)

    X_mean = X_FM_factor @ X_FM_factor.T

    return X_mean

def frechet_variance(X, rank):
    """Compute the Frechet variance of a set of correlation matrices.
    
    Parameters
    ----------
    X : 3D array (n_samples, m * m correlation matrices) representing correlation matrices.
    rank : int representing the parameter of the quotient geometry
    
    Returns
    -------
    X_variance : real number representing the Frechet variance of the correlation matrices."""

    X = check_dimension(X)

    n_samples = X.shape[0]
    X_mean = frechet_mean(X, rank)
    
    X, X_mean = check_base_dim(X, X_mean)
    X_dist = distance(X, X_mean, rank)
    X_variance = np.sum(X_dist ** 2) / n_samples

    return X_variance

def clustering(X, K_groups, rank):
    """Perform clustering on a set of correlation matrices.
    
    Parameters
    ----------
    X : array (n_samples, m * m correlation matrices) representing correlation matrices.
    rank : int representing the parameter of the quotient geometry
    K_groups : int representing the number of clusters
    
    Returns
    -------
    labels : array (n_samples) representing the cluster labels."""

    X = check_dimension(X)

    pairwise_distance_matrix = pairwise_distance(X, X, rank = rank)

    dist_matrix_condensed = pairwise_distance_matrix[np.triu_indices(X.shape[-1], k=1)] 
    clusters = linkage(dist_matrix_condensed, method='ward')
    labels_estimate = (fcluster(clusters, K_groups, criterion='maxclust') - 1).astype(int)

    return labels_estimate

# ------------------- Bures-Wasserstein Geometry -------------------

def distance_bw(X, Y):
    """Compute the distance between two PSD matrices using the Bures-Wasserstein geometry.

    Parameters
    ----------
    X : 2D array (m * m PSD matrix) 
    Y : 2D array (m * m PSD matrix) 

    Returns
    -------
    dist : real number representing the distance between the PSD matrices."""

    return np.sqrt(np.trace(X + Y - 2 * sqrtm(sqrtm(X) @ Y @ sqrtm(X))))

# ------------------- Evaluation -------------------

import itertools

def accuracy(labels_hat, true_labels, align = False):
    labels_hat = labels_hat.astype(int)
    true_labels = true_labels.astype(int)
    K_groups = np.max(labels_hat) + 1
    numbers = list(range(K_groups))
    permutations = list(itertools.permutations(numbers))
    accuracy = np.zeros(len(permutations))
    for j in range(len(permutations)):
        labels = np.zeros(len(labels_hat))
        for i in range(K_groups):
            labels[labels_hat == i] = permutations[j][i]
        accuracy[j] = np.sum(labels == true_labels) / len(labels)
    if align:
        j = np.argmax(accuracy)
        labels = np.zeros(len(labels_hat))
        for i in range(K_groups):
            labels[labels_hat == i] = permutations[j][i]
        return np.max(accuracy), labels
    else:
        return np.max(accuracy)

def rand_score(true_labels, estimated_labels):
    return sklearn.metrics.rand_score(true_labels, estimated_labels)

def adjusted_rand_score(true_labels, estimated_labels):
    return sklearn.metrics.adjusted_rand_score(true_labels, estimated_labels)

def mutual_info_score(true_labels, estimated_labels):
    return sklearn.metrics.mutual_info_score(true_labels, estimated_labels)

def normalized_mutual_info_score(true_labels, estimated_labels):
    return sklearn.metrics.normalized_mutual_info_score(true_labels, estimated_labels)

def adjusted_mutual_info_score(true_labels, estimated_labels):
    return sklearn.metrics.adjusted_mutual_info_score(true_labels, estimated_labels)