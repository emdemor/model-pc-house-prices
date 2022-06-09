from typing import Any
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

TYPE_MAPPING = {
    # -- home -----
    "TWO_STORY_HOUSE": "HOME",
    "CONDOMINIUM": "HOME",
    # -- apartment -----
    "FLAT": "APARTMENT",
    "KITNET": "APARTMENT",
    "PENTHOUSE": "APARTMENT",
    # -- allotment land -----
    "RESIDENTIAL_ALLOTMENT_LAND": "ALLOTMENT_LAND",
    "COMMERCIAL_ALLOTMENT_LAND": "ALLOTMENT_LAND",
    # -- business -----
    "SHED_DEPOSIT_WAREHOUSE": "BUSINESS",
    "BUSINESS": "BUSINESS",
    "COMMERCIAL_PROPERTY": "BUSINESS",
    "COMMERCIAL_BUILDING": "BUSINESS",
    "OFFICE": "BUSINESS",
    "HOTEL": "BUSINESS",
    # -- country -----
    "COUNTRY_HOUSE": "COUNTRY",
    "FARM": "COUNTRY",
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
    data = data.loc[data["type"] != "HOTEL"].copy()

    # -- Terrenos não tem certas propriedades ----------
    data.loc[
        data["type"].str.contains("ALLOTMENT"),
        ["n_parking_spaces", "n_bathrooms", "n_bedrooms"],
    ] = np.nan

    # -- mapping types -----
    data["type"] = data["type"].replace(TYPE_MAPPING)

    # -- generate dummy variables for type ------
    categories = ["APARTMENT", "HOME", "ALLOTMENT_LAND", "COUNTRY"]
    data = generate_dummy_variables(data, "type", categories, prefix="column")

    return data


def generate_dummy_variables(
    dataframe: pd.DataFrame,
    column: str,
    categories: list or tuple or np.array,
    sep: str = "_",
    prefix: Any = "column",
):
    if prefix == "column":
        prefix = column

    for cat in categories:
        if prefix is not None:
            column_name = prefix + sep + cat
        else:
            column_name = cat

        dataframe = dataframe.assign(
            **{column_name: (dataframe[column] == cat).astype("Int8")}
        )

    return dataframe
