import numpy #Import the NumPy library which helps us with array manipulation. Our samples are stored in arrays with one row per sample and one column per variable. NumPy lets us work on all of them at the same time.


def sample_mean_cov(X):
#Returns the sample mean of each variable and the sample covariance matrix. Each row of X is one sample and each column is one variable.

#Converts X into an array and changes the data type to accept decimals
	X = numpy.asarray(X, dtype=float)

#Number of samples, which is the number of rows
	N = X.shape[0]

#We divide by N - 1 below, so there has to be at least two samples
	if N < 2:
		raise ValueError("X must have at least two samples")

#Takes the average of each column, which gives the sample mean of each variable
	mean_vector = numpy.mean(X, axis=0)

#Subtracts the mean from every sample so each column has an average of 0
	X_centered = X - mean_vector

#Multiplies the samples from the last step by themselves to get the covariance matrix. We divide by N - 1 instead of N. The mean came from the same samples, so dividing by N would make the covariance a little too small.
	cov_matrix = (X_centered.T @ X_centered) / (N - 1)

	return mean_vector, cov_matrix


def is_valid_covariance(S, tol=1e-10):
#Returns True if S is a valid covariance matrix, which means it is symmetric and positive semidefinite

#Converts S into an array and changes the data type to accept decimals
	S = numpy.asarray(S, dtype=float)

#A covariance matrix has to be square
	if S.ndim != 2 or S.shape[0] != S.shape[1]:
		return False

#Checks that S is symmetric. Every entry has to match the entry on the other side of the diagonal, within the tolerance.
	if not numpy.allclose(S, S.T, rtol=0, atol=tol):
		return False

#We use numpy.linalg.eigvalsh because S is symmetric which means it always returns the imaginary numbersmake . Since we only want real numbers we use eigvalsh.
	eigenvalues = numpy.linalg.eigvalsh(S)

#Positive semidefinite means every eigenvalue is 0 or larger. The tolerance lets a very small negative eigenvalue pass, since rounding can cause one.
	return bool(numpy.min(eigenvalues) >= -tol)


def gaussian_pdf(x, mu, Sigma):
#Returns the multivariate Gaussian density, built from the determinant and solve

#Converts the inputs into arrays and changes the data type to accept decimals. x can be one point or many points, with one point per row.
	x = numpy.asarray(x, dtype=float)
	mu = numpy.asarray(mu, dtype=float)
	Sigma = numpy.asarray(Sigma, dtype=float)

#Number of variables in each point
	d = len(mu)

#Determinant of Sigma. The density only exists when it is greater than 0.
	det_Sigma = numpy.linalg.det(Sigma)
	if det_Sigma <= 0:
		raise ValueError("Sigma must have a positive determinant")

#Difference between each point and the mean
	diff = x - mu

#We use numpy.linalg.solve because solve is faster and more accurate than using the inverse. Solving finds the answer and skips the extra steps. When sigma is closer to singular using inverse makes it prone to rounding errors which solve avoids. 
	z = numpy.linalg.solve(Sigma, diff.T).T

#Squared Mahalanobis distance. This is how far each point is from the mean, measured in a way that takes the covariance into account.
	mahalanobis_sq = numpy.sum(diff * z, axis=-1)

#Scaling number that makes the density add up to 1. It uses the determinant of Sigma.
	normalizer = 1 / numpy.sqrt((2 * numpy.pi) ** d * det_Sigma)

#Multiplies the scaling number by the exponential of the distance to get the density at each point
	return normalizer * numpy.exp(-0.5 * mahalanobis_sq)


def whitening_matrix(Sigma):
#Returns the whitening matrix W. W changes the samples so each variable has a variance of 1 and no correlation with the others.

#Converts Sigma into an array and changes the data type to accept decimals
	Sigma = numpy.asarray(Sigma, dtype=float)

#We use numpy.linalg.eigh because Sigma is symmetric. Eigh returns real eigenvalues in order from smallest to largest and eigenvectors that are at right angles to each other and have a length of 1. If we used eig it can return eigenvalues containing imaginary numbers.
	eigenvalues, V = numpy.linalg.eigh(Sigma)

#We divide by the square root of each eigenvalue below, so every eigenvalue has to be greater than 0
	if numpy.min(eigenvalues) <= 0:
		raise ValueError("Sigma must be positive definite")

#Builds a diagonal matrix. Each spot on the diagonal is 1 divided by the square root of an eigenvalue.
	lambda_inv_sqrt = numpy.diag(1 / numpy.sqrt(eigenvalues))

#Builds W. When W is applied to the samples, it rotates them to line up with the eigenvectors. Then it scales each direction so its variance becomes 1.
	return lambda_inv_sqrt @ V.T


def lmmse_weights(Sigma_xx, Sigma_sx):
#Returns the weights for the linear MMSE estimator

#Converts both inputs into arrays and changes the data type to accept decimals
	Sigma_xx = numpy.asarray(Sigma_xx, dtype=float)
	Sigma_sx = numpy.asarray(Sigma_sx, dtype=float)

#We use numpy.linalg.solve because solve is faster and more accurate than using the inverse. Solving finds the answer and skips the extra steps. When Sigma_xx is close to singular using inverse makes it prone to errors which solve can help with. 
	return numpy.linalg.solve(Sigma_xx, Sigma_sx)
