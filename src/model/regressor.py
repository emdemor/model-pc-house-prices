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