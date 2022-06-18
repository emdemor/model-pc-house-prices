from xgboost import XGBRegressor
from src.base.commons import load_pickle, load_yaml


def set_regressor(model, **regressor_args):

    model_config = load_yaml(f"config/model.yaml")

    static_parameters = model_config["static_parameters"]

    model = model_config["model"]

    if model == "xgboost":
        model = XGBRegressor(**static_parameters, **regressor_args)
        return model


def get_regressor():
    filepaths = load_yaml(filename="config/filepaths.yaml")
    model_config = load_yaml(f"config/model.yaml")

    regressor = load_pickle(
        filepaths["model_regressor_path"].format(model=model_config["model"])
    )
    return regressor


def get_model_parameters():
    model_config = load_yaml("config/model.yaml")

    model_parameters = {}
    model_parameters["model"] = model_config["model"]
    model_parameters["metric"] = model_config["metric"]
    model_parameters["parametric_space"] = model_config["parametric_space"]

    return model_parameters