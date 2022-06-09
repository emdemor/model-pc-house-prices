import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TARGET_COLUMN = "price"
TARGET_SCALE = "log10"  # 'log', 'log10', 'log1p', 'log101p
TARGET_TRANSFORMATIONS = {
    "log": np.log,
    "log10": np.log10,
    "log1p": np.log1p,
    "log101p": lambda x: np.log10(1 + x),
}


def build_features(data: pd.DataFrame) -> tuple:

    data = clear_dataset(data)

    data = dataset_treatment(data)

    X = data.drop(columns=TARGET_COLUMN, errors="ignore")

    y = transform_target(data[TARGET_COLUMN])

    return X, y


def transform_target(y: pd.Series) -> pd.Series:

    y = TARGET_TRANSFORMATIONS[TARGET_SCALE](y)

    y.name = TARGET_SCALE + "_" + y.name

    return y


def clear_dataset(data: pd.DataFrame) -> pd.DataFrame:

    # remove registers where response variable is null
    clean_data = data.loc[data[TARGET_COLUMN].notna()]

    # remove registers where response variable is zero
    clean_data = data.loc[data[TARGET_COLUMN] > 0]

    return clean_data


def dataset_treatment(data: pd.DataFrame) -> pd.DataFrame:

    # -- Imóveis do tipo Hoteis são muito peculiares --------
    data = data.loc(data["type"] == "HOTEL", errors="ignore")

    # -- Terrenos não tem certas propriedades ----------
    data["n_parking_spaces"] = np.where(
        data["type"].contains("ALLOTMENT"), np.nan, data["n_parking_spaces"]
    )
    data["n_bathrooms"] = np.where(
        data["type"].contains("ALLOTMENT"), np.nan, data["n_bathrooms"]
    )
    data["n_bedrooms"] = np.where(
        data["type"].contains("ALLOTMENT"), np.nan, data["n_bedrooms"]
    )

    return data
