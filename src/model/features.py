import unidecode
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import get_config
import unidecode
from src.base.commons import to_snake_case

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

LONG_INTERVAL = [-46.7, -46.4]

LAT_INTERVAL = [-21.9, -21.76]


def build_features(data: pd.DataFrame) -> tuple:

    data = clear_dataset(data)

    data = dataset_treatment(data)

    data = construct_features(data)

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

    data = treat_type(data)

    data = treat_latlong(data)

    data = treat_neighborhood(data)

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


def construct_features(data: pd.DataFrame) -> pd.DataFrame:

    # data["has_condo_fee"] = np.where(data["condo_fee"] > 0, 1, 0)

    return data


def treat_type(data: pd.DataFrame) -> pd.DataFrame:

    # -- Imóveis do tipo Hoteis são muito peculiares --------
    data = data.loc[data["type"] != "HOTEL"].copy()

    # -- Terrenos não tem certas propriedades ----------
    data.loc[
        data["type"].str.contains("ALLOTMENT"),
        ["n_parking_spaces", "n_bathrooms", "n_bedrooms"],
    ] = np.nan

    # -- Unificando tipos comuns -----
    data["type"] = data["type"].replace(TYPE_MAPPING)

    # -- Gerando variáveis dummy para os tipos ------
    categories = ["APARTMENT", "HOME", "ALLOTMENT_LAND", "COUNTRY"]
    data = generate_dummy_variables(data, "type", categories, prefix="column")

    # -- Anulando os casos onde a área é zero ------------
    data.loc[
        (data["type"].isin(["ALLOTMENT_LAND", "COUNTRY", "BUSINESS"]))
        & (data["area"] == 0),
        ["area"],
    ] = np.nan

    return data


def treat_latlong(data: pd.DataFrame) -> pd.DataFrame:

    # -- Removendo as latitudes e longitudes zeradas ---------
    data.loc[data["longitude"] > -10, "longitude"] = np.nan
    data.loc[data["latitude"] > -10, "latitude"] = np.nan

    # -- tratando os casos onde latitude e longitude estão invertidas
    conditions_lat_long_inverted = data["latitude"].between(*LONG_INTERVAL) & data[
        "longitude"
    ].between(*LAT_INTERVAL)

    values = data.loc[conditions_lat_long_inverted][["latitude", "longitude"]].to_dict(
        orient="records"
    )[0]

    data.loc[conditions_lat_long_inverted, "latitude"] = values["longitude"]
    data.loc[conditions_lat_long_inverted, "longitude"] = values["latitude"]

    # -- Anulando as latitudes e longitudes fora do intervalo -----
    data.loc[~data["latitude"].between(*LAT_INTERVAL), "latitude"] = np.nan
    data.loc[~data["longitude"].between(*LONG_INTERVAL), "longitude"] = np.nan

    return data


def treat_neighborhood(data: pd.DataFrame) -> pd.DataFrame:
    def decode(x):
        if x is not None:
            return unidecode.unidecode(x)
        else:
            return x

    try:
        maps = get_config(filename="config/neighbor_rename.yaml")
    except:
        maps = {}

    data["neighborhood"] = data["neighborhood"].replace(maps)

    data["neighborhood"] = data["neighborhood"].apply(decode)

    data["neighborhood"] = data["neighborhood"].apply(
        lambda x: to_snake_case(x) if x is not None else None
    )

    data.loc[data["neighborhood"] == "", "neighborhood"] = None

    return data
