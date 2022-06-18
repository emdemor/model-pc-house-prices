import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from src.config import get_config
from src.base.commons import dump_json, dump_pickle, load_json, load_yaml
from src.model.regressor import set_regressor


def train_regressor():

    # Getting data information
    model_config = load_yaml("config/model.yaml")

    try:
        model_parameters = load_json(f"model/{model_config['model']}/config.json")

    except:
        for i, param in enumerate(model_config["parametric_space"]):
            param.update({"best_value": param["estimate"]})

        dump_json(model_config, f"model/{model_config['model']}/config.json")

        model_parameters = load_json(f"model/{model_config['model']}/config.json")

    filepaths = get_config(filename="config/filepaths.yaml")

    hyper_param = {
        hp["parameter"]: hp["best_value"] for hp in model_parameters["parametric_space"]
    }

    X_train = pd.read_csv(filepaths["data_train_features_path"])
    y_train = pd.read_csv(filepaths["data_train_target_path"]).iloc[:, 0]
    X_test = pd.read_csv(filepaths["data_test_features_path"])
    y_test = pd.read_csv(filepaths["data_test_target_path"]).iloc[:, 0]

    regressor = set_regressor(model_config["model"], **hyper_param)

    regressor = regressor.__class__(**hyper_param)

    # Treinando para obter a métrica
    regressor.fit(X_train, y_train)

    y_test_pred = regressor.predict(X_test)

    print("mae_test = {:.2f}".format(mean_absolute_error(y_test, y_test_pred)))
    print("R2_test = {:.2f}".format(r2_score(y_test, y_test_pred)))

    # Treinando com o dataset completo
    regressor.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))

    # Export preprocessing artifacts
    dump_pickle(
        regressor, filepaths["model_regressor_path"].format(model=model_config["model"])
    )


if __name__ == "__main__":
    train_regressor()