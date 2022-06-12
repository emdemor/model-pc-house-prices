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


PARAMETERS_CONFIG = get_config(filename="config/parameters.yaml")
TARGET_TRANSFORMATIONS = {
    "log": np.log,
    "log10": np.log10,
    "log1p": np.log1p,
    "log101p": lambda x: np.log10(1 + x),
}


def make_dataset(config: dict, extract_data: bool = False) -> pd.DataFrame:
    if extract_data:
        extract_dataset(config)

    data_basic = pd.read_parquet(config["data_raw_basic_path"])

    # TODO
    # Inserir aqui os dados construidos a partir de
    # 1. Amenities
    # 2. Description
    # 3. Points of Interest
    data = data_basic

    # Remover os registros onde a variável responsta
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
