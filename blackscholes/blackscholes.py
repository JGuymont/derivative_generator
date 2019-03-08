import scipy.stats as stats
import numpy as np


def blackscholes(S, K, T, q, v, r, style):
        d1 = (np.log(S / K) + (r - q + 0.5 * v**2) * T) / (v * np.sqrt(T))
        d2 = d1 - v * np.sqrt(T)
        return style * S * np.exp(-q * T) * stats.norm.cdf(d1 * style) - style * K * np.exp(-r * T) * stats.norm.cdf(d2 * style)
