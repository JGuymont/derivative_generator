import pickle
import pandas


def pickle2dataframe(path: str):
    """
    Load a pickle and convert it to a `pandas.DataFrame`

    Args:
        path ([type]): [description]

    Returns:
        [type]: [description]
    """
    with open(path, 'rb') as f:
        unpickled = pickle.load(f)
    dataframe = pandas.DataFrame(unpickled)
    return dataframe


def dataframe2pickle(dataframe, path):
    with open(path, 'wb') as f:
        pickle.dump(dataframe, f, pickle.HIGHEST_PROTOCOL)
