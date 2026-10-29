import numpy
import scipy


def inv_cdf_exponential(u, lam):
    #Inverse CDF for an exponential random variable

    #Converts the input into a NumPy array so the operation works
    #with either a single value or an entire array of values
    u = numpy.asarray(u, dtype=float)

    #The rate parameter must be positive.
    if lam <= 0:
        raise ValueError("lam must be positive")

    #rng.random() produces values from 0 up to, but not including, 1
    #The inverse CDF is only valid for 0 <= u < 1
    if numpy.any((u < 0) | (u >= 1)):
        raise ValueError("u must satisfy 0 <= u < 1")

    #Inverse CDF of an exponential random variable:
    #F^(-1)(u) = -ln(1-u)/lambda
    return -numpy.log1p(-u) / lam


def inv_cdf_rayleigh(u, sigma):
    #Inverse CDF for a Rayleigh random variable

    #Converts the input into a NumPy array so the operation works
    #with either a single value or an entire array of values
    u = numpy.asarray(u, dtype=float)

    #The scale parameter must be positive
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    #The inverse CDF is valid for 0 <= u < 1
    if numpy.any((u < 0) | (u >= 1)):
        raise ValueError("u must satisfy 0 <= u < 1")

    #Inverse CDF of a Rayleigh random variable:
    #F^(-1)(u) = sigma*sqrt(-2ln(1-u))
    return sigma * numpy.sqrt(-2 * numpy.log1p(-u))


def sample_by_inversion(inv_cdf, n, rng, **kw):
    #Generate samples using uniform random numbers and an inverse CDF

    #Generates n uniform random numbers
    #This is the only source of randomness in this function
    u = rng.random(n)

    #Passes the uniform values into the selected inverse CDF function
    return inv_cdf(u, **kw)


def ks_statistic(samples, cdf):
    #Return the largest gap between empirical and analytic CDFs

    #Sorts the samples because the empirical CDF is calculated
    #from the smallest value to the largest value
    samples = numpy.sort(numpy.asarray(samples, dtype=float))

    #Number of samples
    n = len(samples)

    #Prevents the function from operating on an empty array
    if n == 0:
        raise ValueError("samples cannot be empty")

    #Evaluates the analytic CDF at each sample value
    analytic_cdf = numpy.asarray(cdf(samples), dtype=float)

    #Empirical CDF immediately before each sample
    empirical_before = numpy.arange(n) / n

    #Empirical CDF immediately after each sample
    empirical_after = numpy.arange(1, n + 1) / n

    #Finds the absolute differences between the empirical
    #and analytic CDF values on both sides of every sample
    difference_before = numpy.abs(empirical_before - analytic_cdf)
    difference_after = numpy.abs(empirical_after - analytic_cdf)

    #The KS statistic is the largest difference found.
    return numpy.max(numpy.maximum(difference_before, difference_after))


def q_function(x):
    #Gaussian tail probability Q(x)

    #The Gaussian Q function can be written using the
    #complementary error function: Q(x) = 1/2 erfc(x/sqrt(2))
    #This avoids numerical integration
    return 0.5 * scipy.special.erfc(numpy.asarray(x) / numpy.sqrt(2))

    