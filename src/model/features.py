from src.base import logger
import unidecode
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import get_config
import unidecode
from src.base.commons import to_snake_case

LOGGER = logger.set()

PARAMETERS_CONFIG = get_config(filename="config/parameters.yaml")


def build_features(data: pd.DataFrame) -> pd.DataFrame:

    LOGGER.info("Build features related to state type")
    data = build_type_features(data)

    LOGGER.info("Build features related to date")
    data = build_date_features(data)

    LOGGER.info("Build features related to latitude and longitude")
    data = fill_latlong_by_neighbor(data)

    LOGGER.info("Build features related to distances")
    data = build_distance_features(data)

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
            **{column_name: (dataframe[column] == cat).astype(float)}
        )

    return dataframe


def build_type_features(data: pd.DataFrame) -> pd.DataFrame:

    # -- Gerando variáveis dummy para os tipos ------

    categories = ["APARTMENT", "HOME", "ALLOTMENT_LAND", "COUNTRY"]

    data = generate_dummy_variables(data, "type", categories, prefix="column")

    return data


def build_distance_features(data: pd.DataFrame) -> pd.DataFrame:

    # -- distancia do centro ----------------------------

    center_coords = np.array([-46.567079, -21.790012])

    data["dist_manh"] = np.abs(data["longitude"] - center_coords[0]) + np.abs(
        data["latitude"] - center_coords[1]
    )

    data["dist_square"] = np.square(data["longitude"] - center_coords[0]) + np.square(
        data["latitude"] - center_coords[1]
    )

    data["dist"] = np.sqrt(data["dist_square"])

    return data


def build_date_features(data: pd.DataFrame) -> pd.DataFrame:

    data["search_date"] = list(data["search_date"])

    data["time_delta"] = (
        pd.to_datetime(data["search_date"]) - pd.to_datetime("2021-01-01")
    ).dt.days

    data["year"] = pd.to_datetime(data["search_date"]).dt.year

    data["month"] = pd.to_datetime(data["search_date"]).dt.month

    data["day"] = pd.to_datetime(data["search_date"]).dt.day

    return data


def fill_latlong_by_neighbor(data: pd.DataFrame) -> pd.DataFrame:

    # -- Preenchendo os latlongs faltantes com as coordenadas dos bairros

    try:
        data["longitude"] = np.where(
            data["longitude"].isna(), data["neighbor_longitude"], data["longitude"]
        )
    except Exception as err:
        logging.warning(err)

    try:
        data["latitude"] = np.where(
            data["latitude"].isna(), data["neighbor_latitude"], data["latitude"]
        )
    except Exception as err:
        logging.warning(err)

    return data


def neighbors_one_hot_encode(X: pd.DataFrame, prefix="neighbor_") -> pd.DataFrame:

    selected_neihbors = get_config(
        "model/preprocessing/onehot_encode_neighborhood.yaml"
    )

    list_ = [
        np.where(X["neighborhood"].fillna("") == neighbor, 1, 0)
        for neighbor in selected_neihbors
    ]

    ohe = pd.DataFrame(
        np.array(list_).T,
        columns=[prefix + str(neighbor) for neighbor in selected_neihbors],
        index=X.index,
    )

    return pd.concat([X.drop(columns=["neighborhood"]), ohe], axis=1)
