import pandas as pd


class Preprocessor:
    def __init__(self, categorical_features=None, categorical_transform='onehot', drop_na=False, to_drop=None):
        self.categorical_features = categorical_features
        self.categorical_transform = categorical_transform
        self.drop_na = drop_na
        self.to_drop = to_drop

    def cat2onhot(self, dataframe):
        dataframe = pd.get_dummies(dataframe, columns=self.categorical_features)

    def cat2int(self, dataframe):
        for feature in self.categorical_features:
            dataframe[feature] = pd.Categorical(dataframe[feature]).codes
        return dataframe

    def __preprocess_categorical(self):
        return self.categorical_features and self.categorical_transform

    def preprocess(self, dataframe):

        if self.to_drop:
            dataframe = dataframe.drop(self.to_drop, axis=1)

        if self.drop_na:
            dataframe.dropna(inplace=True)

        if self.__preprocess_categorical():
            if self.categorical_transform == 'onehot':
                return self.cat2onhot(dataframe)
            elif self.categorical_features == 'category':
                return self.cat2int(dataframe)
            else:
                raise ValueError

    def __call__(self, dataframe):
        return self.preprocess(dataframe)
