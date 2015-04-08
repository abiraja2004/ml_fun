#!/usr/bin/env python

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

data = load_iris()
X = data.data

# convert features in column 1 from cm to inches
#X[:,0] /= 2.54
# convert features in column 2 from cm to meters
#X[:,1] /= 100

def pca_raw_cov(X):
    # Compute the covariance matrix
    cov_mat = np.cov(X.T)

    # Eigendecomposition of the covariance matrix
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

    # Make a list of (eigenvalue, eigenvector) tuples
    # and sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs_cov = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
    eig_pairs_cov.sort()
    eig_pairs_cov.reverse()

    # Construct the transformation matrix W from the eigenvalues that correspond to
    # the k largest eigenvalues (here: k = 2)
    matrix_w_cov = np.hstack((eig_pairs_cov[0][1].reshape(4,1), eig_pairs_cov[1][1].reshape(4,1)))

    # Transform the data using matrix W
    X_raw_transf = matrix_w_cov.T.dot(X.T).T

    # Plot the data
    plt.scatter(X_raw_transf[:,0], X_raw_transf[:,1])
    plt.title('PCA based on the covariance matrix of the raw data')
    plt.show()

def pca_standardize_cov(X):

    # Standardize data
    X_std = StandardScaler().fit_transform(X)

    # Compute the covariance matrix
    cov_mat = np.cov(X_std.T)

    # Eigendecomposition of the covariance matrix
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

    # Make a list of (eigenvalue, eigenvector) tuples
    # and sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs_cov = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
    eig_pairs_cov.sort()
    eig_pairs_cov.reverse()

    # Construct the transformation matrix W from the eigenvalues that correspond to
    # the k largest eigenvalues (here: k = 2)
    matrix_w_cov = np.hstack((eig_pairs_cov[0][1].reshape(4,1), eig_pairs_cov[1][1].reshape(4,1)))

    # Transform the data using matrix W
    X_std_transf = matrix_w_cov.T.dot(X_std.T).T

    # Plot the data
    plt.scatter(X_std_transf[:,0], X_std_transf[:,1])
    plt.title('PCA based on the covariance matrix after standardizing the data')
    plt.show()

def pca_cor(X):

    # Standardize data
    X_std = StandardScaler().fit_transform(X)

    # Compute the correlation matrix
    cor_mat = np.corrcoef(X.T)

    # Eigendecomposition of the correlation matrix
    eig_val_cor, eig_vec_cor = np.linalg.eig(cor_mat)

    # Make a list of (eigenvalue, eigenvector) tuples
    # and sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs_cor = [(np.abs(eig_val_cor[i]), eig_vec_cor[:,i]) for i in range(len(eig_val_cor))]
    eig_pairs_cor.sort()
    eig_pairs_cor.reverse()

    # Construct the transformation matrix W from the eigenvalues that correspond to
    # the k largest eigenvalues (here: k = 2)
    matrix_w_cor = np.hstack((eig_pairs_cor[0][1].reshape(4,1), eig_pairs_cor[1][1].reshape(4,1)))

    # Transform the data using matrix W
    X_transf = matrix_w_cor.T.dot(X_std.T).T

    # Plot the data
    plt.scatter(X_transf[:,0], X_transf[:,1])
    plt.title('PCA based on the correlation matrix of the raw data')
    plt.show()

from sklearn.decomposition import PCA

def pca_scikit(X):

    # Standardize
    X_std = StandardScaler().fit_transform(X)

    # PCA
    sklearn_pca = PCA(n_components=2)
    X_transf = sklearn_pca.fit_transform(X_std)

    # Plot the data
    plt.scatter(X_transf[:,0], X_transf[:,1])
    plt.title('PCA via scikit-learn (using SVD)')
    plt.show()

pca_raw_cov(X)
pca_standardize_cov(X)
pca_cor(X)
pca_scikit(X)

