import numpy #Import the NumPy library which helps us with array manipulation. since we are creating simulating packets as 4 bits it can be treated as an array. NumPy helps us store the bits into an array and manipulate them all at the same time.


def bsc(bits, p, rng):
	"""Flip each bit independently probability p"""

#Converts list of bits into an array using numpy.asarray
	bits = numpy.asarray(bits) 

#Determines arrays dimension -> generates a random number for each bit, if the random number for that bit is less than p it will be considered true
	flips = rng.random(bits.shape) < p 

#Compares the original array of bits to the array of flip true/false values. False keeps the bit, true flips it. Returns the new array of bits.
	return numpy.logical_xor(bits, flips).astype(bits.dtype)


def packet_errors(n_pkt, n_bits, p, rng):
	"""Return the number of errors in each packet"""

#Uses the binomial model to generate error probablity for each number of errors in the packet. RNG looks at the probability for number of error and selects a how many errors are in that packet. 
	errors = rng.binomial(n_bits, p, size=n_pkt)

	return errors


def attempts_until_clean(n_trials, q, rng):
	"""Attempts until the first clean packet"""
	"""q = per - attempt  clean probability"""

#Generates the number of attempts needed for each trial based on the probability of it happening
	attempts = rng.geometric(q, size=n_trials)

	return attempts


def tv_distance(pmf1, pmf2):
	"""Total Variation distance, both arrays over the same support"""

#Converts both PMFs into an array and changes data type to accept decimals
	pmf1 = numpy.asarray(pmf1, dtype=float)
	pmf2 = numpy.asarray(pmf2, dtype=float)

#Checks that both PMFs use the same support
	if pmf1.shape != pmf2.shape:
		raise ValueError("The PMFs must have the same shape.")

#Absolute difference between binomial and poisson probabilities
	absolute_differences = numpy.abs(pmf1 - pmf2)

#Sum the differences and multiply by 1/2 according to Step 5 of the manual. This helps prevent the issue when adding two probabilties summing to 2. 
	return 0.5 * numpy.sum(absolute_differences)
