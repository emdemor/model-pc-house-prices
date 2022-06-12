from src.base.logger import logging
import os
import pandas as pd
import numpy as np

from src.base.commons import to_snake_case
from src.base.file import read_file_string, download_file
from src.model import __version__

from dotenv import load_dotenv
from sqlalchemy import create_engine
from basix.parquet import write as to_parquet


def extract_dataset(config: dict, download_bases: bool = False):

    if download_bases:
        logging.info("Download basic features")
        extract_scrapped_relational_data(config)

        logging.info("Download amenities features")
        extract_scrapped_amenities_data(config)

        logging.info("Download descriptions")
        extrac_scrapped_description_data(config)

    data = pd.read_parquet(config["data_raw_basic_path"])

    assert data is not None

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


def extrac_scrapped_description_data(config: dict):

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
