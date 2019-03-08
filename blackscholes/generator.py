import numpy as np
import pandas as pd

from .blackscholes import blackscholes


def generate_options(n, optStyle):
    data = {}
    data['S'] = np.random.uniform(0, 100, n)
    data['K'] = data['S'] * np.random.uniform(0.5, 1.5, n)
    data['T'] = np.random.uniform(0, 2, n)
    data['r'] = np.random.uniform(0, 0.2, n)
    data['sigma'] = np.random.uniform(0, 1, n)
    data['q'] = np.random.uniform(0, 0.2, n)

    data['P'] = blackscholes(**data, style=np.ones((n,)) * optStyle)
    data['OS'] = optStyle

    return pd.DataFrame.from_dict(data)
