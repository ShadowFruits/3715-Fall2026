import numpy #Import the NumPy library which helps us with array manipulation. Our samples are stored as arrays with one row per sample and one column per variable, and NumPy lets us work on all of them at the same time with matrix operations.


def sample_mean_cov(X):
#Return the sample mean vector and the sample covariance matrix, X is (N, d)

#Converts X into an array and changes the data type to accept decimals. Each row is one sample and each column is one variable.
	X = numpy.asarray(X, dtype=float)

#Number of samples (rows)
	N = X.shape[0]

#We divide by N - 1 below, so there has to be at least two samples
	if N < 2:
		raise ValueError("X must have at least two samples")

#Takes the average of each column, which gives the sample mean of each variable
	mean_vector = numpy.mean(X, axis=0)

#Subtracts the mean from every sample so each column is centered at 0
	X_centered = X - mean_vector

#Multiplies the centered samples by themselves to get the sample covariance matrix. We divide by N - 1 instead of N because the mean was also found from the same samples, and dividing by N would make the covariance come out a little too small.
	cov_matrix = (X_centered.T @ X_centered) / (N - 1)

	return mean_vector, cov_matrix


def is_valid_covariance(S, tol=1e-10):
#Return True if S is symmetric and positive semidefinite

#Converts S into an array and changes the data type to accept decimals
	S = numpy.asarray(S, dtype=float)

#A covariance matrix has to be square
	if S.ndim != 2 or S.shape[0] != S.shape[1]:
		return False

#Checks that S is symmetric, every entry has to equal the entry across the diagonal (within the tolerance)
	if not numpy.allclose(S, S.T, rtol=0, atol=tol):
		return False

#Finds the eigenvalues. eigvalsh is used instead of eig because S is symmetric, so the eigenvalues are real.
	eigenvalues = numpy.linalg.eigvalsh(S)

#Positive semidefinite means every eigenvalue is 0 or larger. The tolerance allows a slightly negative eigenvalue that comes from rounding.
	return bool(numpy.min(eigenvalues) >= -tol)


def gaussian_pdf(x, mu, Sigma):
#Multivariate Gaussian density built from the determinant and a linear solve

#Converts the inputs into arrays and changes the data type to accept decimals. x can be one point shaped (d,) or many points shaped (N, d).
	x = numpy.asarray(x, dtype=float)
	mu = numpy.asarray(mu, dtype=float)
	Sigma = numpy.asarray(Sigma, dtype=float)

#Dimension of the random vector
	d = len(mu)

#Determinant of Sigma, the density only exists when it is greater than 0
	det_Sigma = numpy.linalg.det(Sigma)
	if det_Sigma <= 0:
		raise ValueError("Sigma must have a positive determinant")

#Difference between each point and the mean
	diff = x - mu

#Uses solve instead of computing the inverse of Sigma directly. solve is faster and more accurate than inv, especially when Sigma is close to singular.
	z = numpy.linalg.solve(Sigma, diff.T).T

#Squared Mahalanobis distance, how far each point is from the mean after accounting for the covariance
	mahalanobis_sq = numpy.sum(diff * z, axis=-1)

#Normalizing constant from the determinant so the density integrates to 1
	normalizer = 1 / numpy.sqrt((2 * numpy.pi) ** d * det_Sigma)

#Multiplies the normalizing constant by the exponential of the distance to get the density at each point
	return normalizer * numpy.exp(-0.5 * mahalanobis_sq)


def whitening_matrix(Sigma):
#Return the whitening matrix W that turns the covariance into the identity matrix

#Converts Sigma into an array and changes the data type to accept decimals
	Sigma = numpy.asarray(Sigma, dtype=float)

#Eigenvalues and eigenvectors (columns of V). eigh is used instead of eig because Sigma is symmetric, so it returns real eigenvalues in ascending order and orthonormal eigenvectors.
	eigenvalues, V = numpy.linalg.eigh(Sigma)

#We divide by the square root of each eigenvalue below, so every eigenvalue has to be greater than 0
	if numpy.min(eigenvalues) <= 0:
		raise ValueError("Sigma must be positive definite")

#Diagonal matrix with 1 divided by the square root of each eigenvalue on the diagonal
	lambda_inv_sqrt = numpy.diag(1 / numpy.sqrt(eigenvalues))

#Rotates the samples onto the eigenvectors and then scales each direction so its variance becomes 1
	return lambda_inv_sqrt @ V.T


def lmmse_weights(Sigma_xx, Sigma_sx):
#Return the linear MMSE weight vector

#Converts both inputs into arrays and changes the data type to accept decimals
	Sigma_xx = numpy.asarray(Sigma_xx, dtype=float)
	Sigma_sx = numpy.asarray(Sigma_sx, dtype=float)

#Uses solve instead of computing the inverse of Sigma_xx directly. solve is faster and more accurate than inv, especially when Sigma_xx is close to singular.
	return numpy.linalg.solve(Sigma_xx, Sigma_sx)
