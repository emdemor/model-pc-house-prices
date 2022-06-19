import pandas as pd
import numpy as np

from src.base import logger
from src.config import get_config
from src.base.commons import dump_pickle
from src.model.features import build_features
from src.model.preprocessing import PreProcessor, apply_preprocess
from src.model.data import (
    get_train_dataset,
    add_external_data,
    export_train_test_datasets,
    sanitize_features,
)

from sklearn.model_selection import train_test_split

LOGGER = logger.set()


def train_preprocessor(extract_data=False):

    LOGGER.info("FUNCTION: train_preprocessor")

    LOGGER.info("Getting information from yamls")
    data_config = get_config(filename="config/filepaths.yaml")
    features_config = get_config(filename="config/features.yaml")

    LOGGER.info("Contructing datasets from raw database")
    raw_scrapped_features, y = get_train_dataset(data_config, extract_data=extract_data)
    raw_scrapped_features = add_external_data(raw_scrapped_features, data_config)

    LOGGER.info("Building features")
    X = build_features(raw_scrapped_features)

    LOGGER.info("Splitting in train and test")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    LOGGER.info("Train preprocessor")
    preprocessor = PreProcessor(features_config)
    preprocessor.fit(X_train, y_train)

    LOGGER.info("Tranforming features")
    X_train = apply_preprocess(preprocessor, X_train)
    X_test = apply_preprocess(preprocessor, X_test)

    LOGGER.info("Exporting splitted dataset")
    export_train_test_datasets(X_train, X_test, y_train, y_test)

    LOGGER.info("Export preprocessor artifacts")
    dump_pickle(preprocessor, data_config["model_preprocessor_path"])


if __name__ == "__main__":
    train_preprocessor()