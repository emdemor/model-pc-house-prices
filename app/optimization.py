import pandas as pd
import numpy as np
import yaml

from src.config import get_config
from src.base.commons import dump_json, dump_pickle, load_json, load_yaml
from src.model.regressor import set_regressor
from src.optimizer import gaussian_process_optimization


from sklearn.metrics import mean_absolute_error, r2_score
from src.model.regressor import set_regressor


def optimize_regressor():

    model_config = load_yaml("config/model.yaml")

    try:
        model_parameters = load_json(f"model/{model_config['model']}/config.json")

    except:
        for i, param in enumerate(model_config["parametric_space"]):
            param.update({"best_value": param["estimate"]})

        dump_json(model_config, f"model/{model_config['model']}/config.json")

        model_parameters = load_json(f"model/{model_config['model']}/config.json")

    hyper_param = {
        hp["parameter"]: hp["best_value"] for hp in model_parameters["parametric_space"]
    }

    filepaths = load_yaml(filename="config/filepaths.yaml")

    X_train = pd.read_csv(filepaths["data_train_features_path"])
    y_train = pd.read_csv(filepaths["data_train_target_path"]).iloc[:, 0]
    X_test = pd.read_csv(filepaths["data_test_features_path"])
    y_test = pd.read_csv(filepaths["data_test_target_path"]).iloc[:, 0]

    optimizer = gaussian_process_optimization(X_train, y_train)

    proposal_params = dict(zip(hyper_param.keys(), optimizer.x))

    proposal_regessor = set_regressor(model_config["model"], **proposal_params)

    proposal_regessor.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)

    y_test_pred = proposal_regessor.predict(X_test)

    mae_proposal = mean_absolute_error(y_test, y_test_pred)
    r2_proposal = r2_score(y_test, y_test_pred)

    print("mae_test = {:.2f}".format(mae_proposal))
    print("R2_test = {:.2f}".format(r2_proposal))

    mae_current = model_parameters["metric"]["value"]

    print("mae_current = ", mae_current)
    print("mae_proposal = ", mae_proposal)

    if mae_proposal < mae_current:

        # atualizar model config
        for i, param in enumerate(model_parameters["parametric_space"]):
            param.update({"best_value": optimizer.x[i]})

        model_parameters["metric"].update({"value": mae_proposal})

        dump_json(
            model_parameters, f"model/{model_parameters['model']}/config.json", indent=4
        )


if __name__ == "__main__":
    optimize_regressor()