import unidecode
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import get_config
import unidecode
from src.base.commons import to_snake_case


PARAMETERS_CONFIG = get_config(filename="config/parameters.yaml")


def build_features(data: pd.DataFrame) -> tuple:

    data = dataset_treatment(data)

    data = construct_features(data)

    data = build_date_features(data)

    return data


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
            **{column_name: (dataframe[column] == cat).astype(float)}
        )

    return dataframe


def construct_features(data: pd.DataFrame) -> pd.DataFrame:

    # data["has_condo_fee"] = np.where(data["condo_fee"] > 0, 1, 0)

    return data


def treat_type(data: pd.DataFrame) -> pd.DataFrame:

    # -- Terrenos não tem certas propriedades ----------
    data.loc[
        data["type"].str.contains("ALLOTMENT"),
        ["n_parking_spaces", "n_bathrooms", "n_bedrooms"],
    ] = np.nan

    # -- Unificando tipos comuns -----
    data["type"] = data["type"].replace(PARAMETERS_CONFIG["TYPE_MAPPING"])

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

    center_coords = np.array([-46.567079, -21.790012])

    # -- Removendo as latitudes e longitudes zeradas ---------
    data.loc[data["longitude"] > -10, "longitude"] = np.nan
    data.loc[data["latitude"] > -10, "latitude"] = np.nan

    # -- tratando os casos onde latitude e longitude estão invertidas
    conditions_lat_long_inverted = data["latitude"].between(
        *PARAMETERS_CONFIG["LONG_INTERVAL"]
    ) & data["longitude"].between(*PARAMETERS_CONFIG["LAT_INTERVAL"])

    if conditions_lat_long_inverted.sum() > 0:
        values = data.loc[conditions_lat_long_inverted][
            ["latitude", "longitude"]
        ].to_dict(orient="records")[0]

        data.loc[conditions_lat_long_inverted, "latitude"] = values["longitude"]
        data.loc[conditions_lat_long_inverted, "longitude"] = values["latitude"]

    # -- Anulando as latitudes e longitudes fora do intervalo -----
    data.loc[
        ~data["latitude"].between(*PARAMETERS_CONFIG["LAT_INTERVAL"]), "latitude"
    ] = np.nan
    data.loc[
        ~data["longitude"].between(*PARAMETERS_CONFIG["LONG_INTERVAL"]), "longitude"
    ] = np.nan

    # -- distancia do centro ----------------------------
    data["dist_manh"] = np.abs(data["longitude"] - center_coords[0]) + np.abs(
        data["latitude"] - center_coords[1]
    )

    data["dist_square"] = np.square(data["longitude"] - center_coords[0]) + np.square(
        data["latitude"] - center_coords[1]
    )

    data["dist"] = np.sqrt(data["dist_square"])

    return data


def treat_neighborhood(data: pd.DataFrame) -> pd.DataFrame:
    def decode(x):
        try:
            return unidecode.unidecode(x)
        except:
            return x

    try:
        maps = get_config(filename="config/neighbor_rename.yaml")
    except:
        maps = {}

    data["neighborhood"] = data["neighborhood"].str.strip()

    data["neighborhood"] = data["neighborhood"].replace(maps)

    data["neighborhood"] = data["neighborhood"].apply(decode)

    def convert_to_snake_case(x):
        try:
            return to_snake_case(x)
        except:
            return x

    data["neighborhood"] = data["neighborhood"].apply(convert_to_snake_case)

    data.loc[data["neighborhood"] == "", "neighborhood"] = None

    data.loc[
        data["neighborhood"].fillna("").str.contains("chacara"), "neighborhood"
    ] = "zona_rural"

    # data = pd.concat(
    #     [
    #         data.drop(columns="neighborhood"),
    #         pd.get_dummies(data["neighborhood"], prefix="neighbor"),
    #     ],
    #     axis=1,
    # )

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
