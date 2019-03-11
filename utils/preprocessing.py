import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Preprocessor:
    """
    Preprocessing function

    Arguments
        categorical_features: list or None, default: None
            List of categorical features
        categorical_transform: str {'onehot', 'category', None}, default: 'onehot'
            Transformation to apply to the categorical features. 'onehot' means that
            the categorical features are transform in one-hot vector. 'category' means
            that an integer is assigned to each the category.
        to_drop: list or None, default: None
            List of the name of the features to drop

    Raises:
        ValueError: the value of categorical_transform is not valid

    Returns:
        pandas.DataFrame: the preprocessed dataframe
    """

    def __init__(self, categorical_features, numerical_features):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.all_features = categorical_features + numerical_features

    def cat2onhot(self, dataframe):
        dataframe = pd.get_dummies(dataframe, columns=self.categorical_features)
        return dataframe

    def cat2int(self, dataframe):
        for feature in self.categorical_features:
            dataframe[feature] = pd.Categorical(dataframe[feature]).codes
        return dataframe

    def __preprocess_categorical(self):
        return self.categorical_features and self.categorical_transform

    def preprocess_(self, dataframe):

        dataframe = dataframe[self.all_features]

        if self.__preprocess_categorical():
            if self.categorical_transform == 'onehot':
                return self.cat2onhot(dataframe)
            elif self.categorical_features == 'category':
                return self.cat2int(dataframe)
            else:
                raise ValueError
        return dataframe

    def preprocess(self, data):
        data = data[self.all_features]
        label_encoders = {}
        for feature in self.categorical_features:
            label_encoders[feature] = LabelEncoder()
            data.loc[feature] = label_encoders[feature].fit_transform(data[feature])
        return data

    def __call__(self, dataframe):
        return self.preprocess(dataframe)
