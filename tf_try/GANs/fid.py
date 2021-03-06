# coding: utf-8
''' Calculates the Frechet Inception Distance (FID) to evalulate GANs.

The FID metric calculates the distance between two distributions of dataset.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
'''

import numpy as np
from scipy import linalg


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
       mu1 : Numpy array for generated samples.
       mu2   : The sample mean precalcualted on an representive data set.
       sigma1: The covariance matrix for generated samples.
       sigma2: The covariance matrix precalcualted on an representive data set.

    Returns:
       : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    assert diff.dot(diff) >= 0
    assert np.trace(sigma1) >= 0, '应该大于0'
    assert np.trace(sigma2) >= 0
    assert tr_covmean >= 0
    """print('begin')
    print(diff.dot(diff))
    print(np.trace(sigma1))
    print(np.trace(sigma2))
    print(tr_covmean)
    print(np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    print('end')"""
    if diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean < 0:
        pass
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def calculate_statistics(data):
    """Calculation of the statistics used by the FID.
    Params:
       data: Original data.
    Returns:
       mu    : The mean over samples.
       sigma : The covariance matrix of the samples.
    """
    mu = np.mean(data, axis=0)
    sigma = np.cov(data, rowvar=False)
    return mu, sigma