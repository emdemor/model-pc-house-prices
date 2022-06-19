import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from src.config import get_config
from src.base import logger
from src.base.commons import dump_json, dump_pickle, load_json, load_yaml
from src.model.regressor import set_regressor

LOGGER = logger.set()


def train_regressor():

    LOGGER.info("FUNCTION: train_regressor")

    LOGGER.info("Getting parameter information")

    # Getting data information
    model_config = load_yaml("config/model.yaml")

    # Importando os parâmetros do modelo
    model_parameters = load_json(model_config["parametric_space_path"])

    # Montando o dicionário de parâmetros
    hyper_param = {
        hp["parameter"]: hp["estimate"] for hp in model_parameters["parametric_space"]
    }

    filepaths = get_config(filename="config/filepaths.yaml")

    hyper_param = {
        hp["parameter"]: hp["best_value"] for hp in model_parameters["parametric_space"]
    }

    LOGGER.info("Importing train and test feature and targets")
    X_train = pd.read_csv(filepaths["data_train_features_path"])
    y_train = pd.read_csv(filepaths["data_train_target_path"]).iloc[:, 0]
    X_test = pd.read_csv(filepaths["data_test_features_path"])
    y_test = pd.read_csv(filepaths["data_test_target_path"]).iloc[:, 0]

    LOGGER.info(
        "Set regressor with parameters in {}".format(
            model_parameters["parametric_space"]
        )
    )
    regressor = set_regressor(model_config["model"], **hyper_param)

    regressor = regressor.__class__(**hyper_param)

    LOGGER.info("Training model with training dataset")
    regressor.fit(X_train, y_train)

    y_test_pred = regressor.predict(X_test)

    LOGGER.info("mae_test = {:.2f}".format(mean_absolute_error(y_test, y_test_pred)))
    LOGGER.info("R2_test = {:.2f}".format(r2_score(y_test, y_test_pred)))

    LOGGER.info("Training model with full dataset")
    regressor.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))

    LOGGER.info("Exporting model artifacts")
    dump_pickle(
        regressor, filepaths["model_regressor_path"].format(model=model_config["model"])
    )


if __name__ == "__main__":
    train_regressor()