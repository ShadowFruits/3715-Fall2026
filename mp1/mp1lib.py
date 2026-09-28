import numpy as np


def bsc(bits, p, rng):
    """Flip each bit independently with probability p."""
    """ Bits = 0/1 integers
        p = probablity factor
        rng = np.random.Generator

        returns an array of bits same shape as bits
    """
    bits = np.asarray(bits)
    flips = rng.random(bits.shape) < p
    return np.logical_xor(bits, flips).astype(bits.dtype)


def packet_errors(n_pkt, n_bits, p, rng):
    """Return the number of errors in each packet."""
    return rng.binomial(n_bits, p, size=n_pkt)


def attempts_until_clean(n_trials, q, rng):
    """Return attempts until the first clean packet.
        n_trials = number of independent packets
        q = per attempt probablity of a clean packet
    """
    return rng.geometric(q, size=n_trials)



def tv_distance(pmf1, pmf2):
    """Total variation distance: 0.5 * sum_k |pmf1[k] - pmf2[k]|.

    Both arrays must be over the same support (same length, same k values).
    """
    pmf1 = np.asarray(pmf1, dtype=float)
    pmf2 = np.asarray(pmf2, dtype=float)
    if pmf1.shape != pmf2.shape:
        raise ValueError("pmfs must share the same support (same length)")
    return 0.5 * np.sum(np.abs(pmf1 - pmf2))