import pandas as pd
import numpy as np

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


def train_preprocessor(extract_data=False):

    # Getting data information
    data_config = get_config(filename="config/filepaths.yaml")
    features_config = get_config(filename="config/features.yaml")

    # Construct the dataset
    raw_scrapped_features, y = get_train_dataset(data_config, extract_data=extract_data)
    raw_scrapped_features = add_external_data(raw_scrapped_features, data_config)

    X = build_features(raw_scrapped_features)

    # Splitting in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Train preprocessor
    preprocessor = PreProcessor(features_config)
    preprocessor.fit(X_train, y_train)

    # Tranforming features
    X_train = apply_preprocess(preprocessor, X_train)
    X_test = apply_preprocess(preprocessor, X_test)

    # Export splitted dataset
    export_train_test_datasets(X_train, X_test, y_train, y_test)

    # Export preprocessing artifacts
    dump_pickle(preprocessor, data_config["model_preprocessor_path"])


if __name__ == "__main__":
    train_preprocessor()