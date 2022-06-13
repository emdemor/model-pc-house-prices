from src.base.logger import logging
import os
import pandas as pd
import numpy as np

from src.base.commons import to_snake_case
from src.base.file import read_file_string, download_file
from src.config import get_config
from src.model import __version__

from dotenv import load_dotenv
from sqlalchemy import create_engine
from basix.parquet import write as to_parquet

import unidecode


PARAMETERS_CONFIG = get_config(filename="config/parameters.yaml")
TARGET_TRANSFORMATIONS = {
    "identity": lambda x: x,
    "log": np.log,
    "log10": np.log10,
    "log1p": np.log1p,
    "log101p": lambda x: np.log10(1 + x),
}


def get_train_dataset(config: dict, extract_data: bool = False) -> pd.DataFrame:
    if extract_data:
        extract_dataset(config)

    data_basic = pd.read_parquet(config["data_raw_basic_path"])
    data_basic = sanitize_features(data_basic)

    # TODO
    # Inserir aqui os dados construidos a partir de
    # 1. Amenities
    # 2. Description
    # 3. Points of Interest
    data = data_basic  # .merge(df_neighbor, on="neighborhood", how="left")

    # Remover os registros onde a variável resposta
    # é nula (ou zero, no caso de preço)
    data = remove_invalid_registers(data)

    X = data.drop(columns=[PARAMETERS_CONFIG["TARGET_COLUMN"]], errors="ignore")

    if PARAMETERS_CONFIG["TARGET_COLUMN"] in data.columns:
        y = transform_target(data[PARAMETERS_CONFIG["TARGET_COLUMN"]])
    else:
        y = None

    return X, y


def transform_target(y: pd.Series) -> pd.Series:

    y = TARGET_TRANSFORMATIONS[PARAMETERS_CONFIG["TARGET_SCALE"]](y)

    y.name = PARAMETERS_CONFIG["TARGET_SCALE"] + "_" + y.name

    return y


def extract_dataset(config: dict) -> None:

    try:
        logging.info("Download basic features")
        extract_scrapped_relational_data(config)

        logging.info("Download amenities features")
        extract_scrapped_amenities_data(config)

        logging.info("Download descriptions")
        extract_scrapped_description_data(config)

    except Exception as err:
        logging.error(err)
        raise Exception


def remove_invalid_registers(data: pd.DataFrame) -> pd.DataFrame:

    if PARAMETERS_CONFIG["TARGET_COLUMN"] in data.columns:

        # remove registers where response variable is null
        data = data.loc[data[PARAMETERS_CONFIG["TARGET_COLUMN"]].notna()]

        # remove registers where response variable is zero
        data = data.loc[data[PARAMETERS_CONFIG["TARGET_COLUMN"]] > 0]

    # -- Imóveis do tipo Hoteis são muito peculiares --------
    data = data.loc[data["type"] != "HOTEL"].copy()

    return data


def extract_scrapped_relational_data(config: dict):

    assert load_dotenv()

    engine = create_engine(
        "postgresql://{user}:{password}@{host}:{port}/{database}".format(
            user=os.getenv("DB_USERNAME"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
        )
    )

    df_basic = pd.read_sql(
        """
        with tab as (
            select
                id
                , search_id
                , search_date 
                , id_zap
                , type
                , n_parking_spaces
                , n_bathrooms
                , n_bedrooms
                , area
                , n_floors
                , units_on_floor
                , n_suites
                , state
                , city
                , neighborhood
                , street
                , longitude
                , latitude
                , condo_fee
                , iptu
                , resale
                , buildings
                , plan_only
                , price
                --, amenities
                --, pois_list
                --, link
                --, description       
            from pocos_de_caldas.imoveis i
        )
        select * from tab t
    """,
        engine,
    )

    df_basic["search_date"] = df_basic["search_date"].dt.date

    to_parquet(
        df_basic,
        config["data_raw_basic_path"],
        overwrite=True,
        partition_cols=["search_date"],
    )


def extract_scrapped_amenities_data(config: dict):

    assert load_dotenv()

    engine = create_engine(
        "postgresql://{user}:{password}@{host}:{port}/{database}".format(
            user=os.getenv("DB_USERNAME"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
        )
    )

    df_amenities = pd.read_sql(
        """
        select
            search_id
            , search_date
            , jsonb_array_elements(amenities) as amenity
        from pocos_de_caldas.imoveis
    """,
        engine,
    )

    df_amenities["search_date"] = df_amenities["search_date"].dt.date

    to_parquet(
        df_amenities,
        config["data_raw_amenities_path"],
        overwrite=True,
        partition_cols=["search_date"],
    )


# def data_extraction_poi(config: dict):

#     assert load_dotenv()

#     engine = create_engine(
#         "postgresql://{user}:{password}@{host}:{port}/{database}".format(
#             user=os.getenv("DB_USERNAME"),
#             password=os.getenv("DB_PASSWORD"),
#             host=os.getenv("DB_HOST"),
#             port=os.getenv("DB_PORT"),
#             database=os.getenv("DB_NAME"),
#         )
#     )

#     df_poi = pd.read_sql(
#         """
#         with tab as (
#             select
#                 id
#                 , search_id
#                 , search_date
#                 , id_zap
#                 , type
#                 --, amenities
#                 , pois_list
#                 --, link
#                 --, description
#             from pocos_de_caldas.imoveis i
#         )
#         select * from tab t
#     """,
#         engine,
#     )

#     df_poi["search_date"] = df_poi["search_date"].dt.date

#     to_parquet(
#         df_poi,
#         config["data_raw_poi_path"],
#         overwrite=True,
#         partition_cols=["search_date"],
#     )


def extract_scrapped_description_data(config: dict):

    assert load_dotenv()

    engine = create_engine(
        "postgresql://{user}:{password}@{host}:{port}/{database}".format(
            user=os.getenv("DB_USERNAME"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
        )
    )

    df_desc = pd.read_sql(
        """
        with tab as (
            select
                search_id
                , search_date
                , description       
            from pocos_de_caldas.imoveis i
        )
        select * from tab t
    """,
        engine,
    )

    df_desc["search_date"] = df_desc["search_date"].dt.date

    to_parquet(
        df_desc,
        config["data_raw_descriptions_path"],
        overwrite=True,
        partition_cols=["search_date"],
    )


def sanitize_features(data: pd.DataFrame) -> pd.DataFrame:

    data = treat_type(data)

    data = treat_latlong(data)

    data = treat_neighborhood(data)

    return data


def treat_type(data: pd.DataFrame) -> pd.DataFrame:

    # -- Terrenos não tem certas propriedades ----------
    data.loc[
        data["type"].str.contains("ALLOTMENT"),
        ["n_parking_spaces", "n_bathrooms", "n_bedrooms"],
    ] = np.nan

    # -- Unificando tipos comuns -----
    data["type"] = data["type"].replace(PARAMETERS_CONFIG["TYPE_MAPPING"])

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

    return data


def add_neighbor_region(data: pd.DataFrame, config: dict) -> pd.DataFrame:

    # Import neighbor region information
    df_neighbor = pd.read_csv(config["data_external_neighbor_path"])
    df_neighbor = df_neighbor.loc[~df_neighbor["neighborhood"].duplicated()]
    data = data.merge(df_neighbor, on="neighborhood", how="left")

    return data


def add_region_population(data: pd.DataFrame, config: dict) -> pd.DataFrame:

    # Import neighbor region information
    df_pop = pd.read_csv(config["data_external_region_pop_path"])
    df_pop = df_pop.loc[~df_pop["neighbor_region"].duplicated()]
    data = data.merge(df_pop, on="neighbor_region", how="left")

    return data


def add_min_income_pct(data: pd.DataFrame, config: dict) -> pd.DataFrame:

    # Import neighbor region information
    df_minc = pd.read_csv(config["data_external_region_min_income_pct_path"])
    df_minc = df_minc.loc[~df_minc["neighbor_region"].duplicated()]
    data = data.merge(df_minc, on="neighbor_region", how="left")

    return data


def add_avg_income(data: pd.DataFrame, config: dict) -> pd.DataFrame:

    # Import neighbor region information
    df_inc = pd.read_csv(config["data_external_region_avg_income_path"])
    df_inc = df_inc.loc[~df_inc["neighbor_region"].duplicated()]
    data = data.merge(df_inc, on="neighbor_region", how="left")

    return data


def add_literacy_rate(data: pd.DataFrame, config: dict) -> pd.DataFrame:

    # Import neighbor region information
    df_lit = pd.read_csv(config["data_external_region_literacy_path"])
    df_lit = df_lit.loc[~df_lit["neighbor_region"].duplicated()]
    data = data.merge(df_lit, on="neighbor_region", how="left")

    return data


def add_external_data(data: pd.DataFrame, config: dict) -> pd.DataFrame:

    data = add_neighbor_region(data, config)
    data = add_region_population(data, config)
    data = add_min_income_pct(data, config)
    data = add_avg_income(data, config)
    data = add_literacy_rate(data, config)

    return data