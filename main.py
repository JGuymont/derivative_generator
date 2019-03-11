import pandas
import pickle
from torch.utils.data import DataLoader

import constants
import utils.file_utils as file_utils
from utils.data import OptionDataset
from utils.preprocessing import Preprocessor
from utils.scaler import Scaler
from regressor.mlp import MLP


if __name__ == '__main__':

    preprocessor = Preprocessor(constants.CATEGORICAL_FEATURES, constants.NUMERICAL_FEATURES + [constants.OUTPUT_FEATURE])

    train_data = OptionDataset(
        path=constants.TRAIN_PATH_NO_MISSING,
        categorical_features=constants.CATEGORICAL_FEATURES,
        numerical_features=constants.NUMERICAL_FEATURES,
        output_feature=constants.OUTPUT_FEATURE,
        preprocess=preprocessor,
        transform=None
    )

    cat_dims = [int(train_data.data[col].nunique()) for col in constants.CATEGORICAL_FEATURES]
    emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]

    trainloader = DataLoader(
        train_data,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    mlp = MLP( 
        embedding_dims=emb_dims,
        n_continuous_features=len(constants.NUMERICAL_FEATURES),
        hidden_dim=100,
        output_dim=1,
        n_epochs=200,
        learning_rate=0.001,
        device=constants.DEVICE).to(constants.DEVICE)
    mlp.fit(trainloader)
