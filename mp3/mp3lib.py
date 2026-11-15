import numpy


def sample_mean_cov(X):
    #Return the sample mean vector and the sample covariance matrix

    #Converts the input into a NumPy array so the operation works
    #with X shaped (N, d): N samples (rows) and d variables (columns)
    X = numpy.asarray(X, dtype=float)

    #Number of samples
    N = X.shape[0]

    #We divide by N - 1, so we need at least two samples
    if N < 2:
        raise ValueError("X must have at least two samples")

    #Sample mean of each column
    mean_vector = numpy.mean(X, axis=0)

    #Subtracts the mean from every sample so each column is centered at 0
    X_centered = X - mean_vector

    cov_matrix = (X_centered.T @ X_centered) / (N - 1)

    return mean_vector, cov_matrix


def is_valid_covariance(S, tol=1e-10):
    #Return True if S is symmetric and positive semidefinite

    #Converts the input into a NumPy array
    S = numpy.asarray(S, dtype=float)

    #A covariance matrix has to be square
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        return False

    #Symmetric: S must equal its transpose (within the tolerance)
    if not numpy.allclose(S, S.T, rtol=0, atol=tol):
        return False

    #Positive semidefinite: every eigenvalue must be 0 or larger
    #eigvalsh is used instead of eig because S is symmetric, so the eigenvalues are real
    #The tolerance allows a slightly negative eigenvalue caused by rounding
    eigenvalues = numpy.linalg.eigvalsh(S)

    return bool(numpy.min(eigenvalues) >= -tol)


def gaussian_pdf(x, mu, Sigma):
    #Multivariate Gaussian density built from the determinant and a linear solve

    #Converts the inputs into NumPy arrays
    #x can be one point shaped (d,) or many points shaped (N, d)
    x = numpy.asarray(x, dtype=float)
    mu = numpy.asarray(mu, dtype=float)
    Sigma = numpy.asarray(Sigma, dtype=float)

    #Dimension of the random vector
    d = len(mu)

    #The density only exists when det(Sigma) is greater than 0
    det_Sigma = numpy.linalg.det(Sigma)
    if det_Sigma <= 0:
        raise ValueError("Sigma must have a positive determinant")

    #Difference between each point and the mean
    diff = x - mu

    #Solves Sigma z = (x - mu) instead of computing Sigma^-1 directly
    #solve is faster and more accurate than inv, especially when Sigma is close to singular
    z =numpy.linalg.solve(Sigma, diff.T).T

    mahalanobis_sq = numpy.sum(diff * z, axis=-1)

    normalizer = 1 / numpy.sqrt((2 * numpy.pi) ** d * det_Sigma)

    return normalizer * numpy.exp(-0.5 * mahalanobis_sq)


def whitening_matrix(Sigma):
    #Return the whitening matrix W = Lambda^(-1/2) V^T so that W Sigma W^T = I

    #Converts the input into a NumPy array
    Sigma = numpy.asarray(Sigma, dtype=float)

    #eigh is used instead of eig because Sigma is symmetric
    #It returns real eigenvalues in ascending order and orthonormal eigenvectors (columns of V)
    eigenvalues, V = numpy.linalg.eigh(Sigma)

    #We divide by sqrt(eigenvalue), so every eigenvalue must be greater than 0
    if numpy.min(eigenvalues) <= 0:
        raise ValueError("Sigma must be positive definite")

    lambda_inv_sqrt = numpy.diag(1 / numpy.sqrt(eigenvalues))

    #W = Lambda^(-1/2) V^T
    return lambda_inv_sqrt @ V.T


def lmmse_weights(Sigma_xx, Sigma_sx):

    #Converts the inputs into NumPy arrays
    Sigma_xx = numpy.asarray(Sigma_xx, dtype=float)
    Sigma_sx = numpy.asarray(Sigma_sx, dtype=float)

    #Solves Sigma_xx w = Sigma_sx instead of computing inv(Sigma_xx) @ Sigma_sx
    #solve is faster and more accurate than inv, especially when Sigma_xx is close to singular
    return numpy.linalg.solve(Sigma_xx, Sigma_sx)
