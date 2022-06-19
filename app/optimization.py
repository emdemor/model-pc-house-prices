import pandas as pd
import numpy as np
import yaml

from src.config import get_config
from src.base import logger
from src.base.commons import dump_json, dump_pickle, load_json, load_yaml
from src.model.regressor import get_model_parameters, set_regressor
from src.optimizer import gaussian_process_optimization


from sklearn.metrics import mean_absolute_error, r2_score
from src.model.regressor import set_regressor

LOGGER = logger.set()


def optimize_regressor():

    LOGGER.info("FUNCTION: optimize_regressor")

    LOGGER.info("Get config from `config/model.yaml`")
    model_config = load_yaml("config/model.yaml")

    LOGGER.info("Get model parameters")
    model_parameters = load_json(model_config["parametric_space_path"])

    hyper_param = {
        hp["parameter"]: hp["estimate"] for hp in model_parameters["parametric_space"]
    }

    filepaths = load_yaml(filename="config/filepaths.yaml")

    LOGGER.info("Importing train and test feature and targets")
    X_train = pd.read_csv(filepaths["data_train_features_path"])
    y_train = pd.read_csv(filepaths["data_train_target_path"]).iloc[:, 0]
    X_test = pd.read_csv(filepaths["data_test_features_path"])
    y_test = pd.read_csv(filepaths["data_test_target_path"]).iloc[:, 0]

    LOGGER.info("Gaussian process optimization")
    optimizer = gaussian_process_optimization(X_train, y_train)

    proposal_params = dict(zip(hyper_param.keys(), optimizer.x))

    LOGGER.info(f"Proposal parameters: {proposal_params}")

    LOGGER.info("Set regressor with proposed parameters")
    proposal_regessor = set_regressor(model_config["model"], **proposal_params)
    proposal_regessor.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)
    y_test_pred = proposal_regessor.predict(X_test)
    mae_proposal = mean_absolute_error(y_test, y_test_pred)
    r2_proposal = r2_score(y_test, y_test_pred)

    LOGGER.info("mae_proposed = {:.2f}".format(mae_proposal))
    LOGGER.info("R2_proposed = {:.2f}".format(r2_proposal))

    mae_current = model_parameters["metric"]["value"]

    LOGGER.info("mae_current = {:.3f}".format(mae_current))
    LOGGER.info("mae_proposal = {:.3f}".format(mae_proposal))

    if mae_proposal < mae_current:

        LOGGER.info("Optimization found a better model.")
        LOGGER.info("Update model with new parameters")
        LOGGER.info("Update {} file".format(model_parameters["parametric_space"]))

        for i, param in enumerate(model_parameters["parametric_space"]):
            param.update({"estimate": optimizer.x[i]})

        model_parameters["metric"].update({"value": mae_proposal})

        dump_json(model_parameters, model_config["parametric_space_path"], indent=4)

    else:
        LOGGER.info(
            "Optimization was not capable to find a better model. Keeping with the old model."
        )


if __name__ == "__main__":
    optimize_regressor()