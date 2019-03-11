import torch

DATA_PATH = './data/PSPOption_RAW_Data.pkl'

# Date with missing values removed
DATA_PATH_NO_MISSING = './data/PSPOption_RAW_Data_no_missing.pkl'
TRAIN_PATH_NO_MISSING = './data/PSPOption_RAW_Data_no_missing_TRAIN.pkl'
VALID_PATH_NO_MISSING = './data/PSPOption_RAW_Data_no_missing_VALID.pkl'
TEST_PATH_NO_MISSING = './data/PSPOption_RAW_Data_no_missing_TEST.pkl'

# features
FEATURES_TO_DROP = ['ValuationDate', 'ExpirationDateTime', 'OPT_LAST_UPDATE_TIME', 'PX_MID', 'PX_Theo_BB', 'PX_Theo_Ivol']

ALL_FEATURES = [
    'ValuationDate'
    'PSPInstrumentCategorizationCode',
    'PSPInstrumentCategorizationID',
    'InstrumentDescription',
    'InstrumentDescription2',
    'BloombergTicker',
    'BloombergUndlTicker',
    'OptionStyle',
    'ExerciseStyle',
    'ExpirationDateTime',
    'StrikePrice',
    'StrikePriceCurrencyCode',
    'OPT_LAST_UPDATE_TIME',
    'OPT_UNDL_PX',
    'OPT_FINANCE_RT',
    'OPT_DIV_YIELD',
    'OPT_TIME_TO_MAT',
    'Fx',
    'Volatility_PX_MID',
    'Volatility_PX_LAST',
    'Volatility_BB_BST',
    'Volatility_Ivol',
    'PX_MID',
    'PX_LAST',
    'PX_Theo_BB',
    'PX_Theo_Ivol',
    'Pricing_Source_Used',
    'PX_VOLUME',
    'PX_VOLUME_1D'
]

NUMERICAL_FEATURES = [
    'StrikePrice',
    'OPT_UNDL_PX',
    'OPT_FINANCE_RT',
    'OPT_DIV_YIELD',
    'OPT_TIME_TO_MAT',
    'Fx',
    'Volatility_PX_MID',
    'Volatility_PX_LAST',
    'Volatility_BB_BST',
    'Volatility_Ivol',
    # 'PX_MID',
    # 'PX_LAST',
    # 'PX_Theo_BB',
    # 'PX_Theo_Ivol',
    'PX_VOLUME',
    'PX_VOLUME_1D'
]

CATEGORICAL_FEATURES = [
    #'PSPInstrumentCategorizationCode',
    #'PSPInstrumentCategorizationID',
    'InstrumentDescription',
    #'InstrumentDescription2',
    'BloombergTicker',
    'BloombergUndlTicker',
    'OptionStyle',
    'ExerciseStyle',
    'StrikePriceCurrencyCode',
#    'Pricing_Source_Used',
]

DATE_FEATURES = ['ValuationDate', 'ExpirationDateTime', 'OPT_LAST_UPDATE_TIME']

OUTPUT_FEATURE = 'PX_LAST'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
