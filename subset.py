"""
Create a dataset that only contains the examples without missing values
"""
import utils.file_utils as file_utils

import constants


if __name__ == '__main__':
    data = file_utils.pickle2dataframe(constants.DATA_PATH)
    data.dropna(inplace=True)
    file_utils.dataframe2pickle(data, constants.NO_MISSING_DATA_PATH)
