from src.base.commons import load_json, load_yaml
from src.model.regressor import set_regressor
from src.optimizer.space import eval_parametric_space_dimension
from skopt import gp_minimize
from skopt.utils import use_named_args
from src.optimizer.cross_validation import cross_validate_score
from sklearn.metrics import r2_score, mean_absolute_error


def gaussian_process_optimization(X_train, y_train):

    model_config = load_yaml("config/model.yaml")
    model_parameters = load_json(f"model/{model_config['model']}/config.json")

    hyper_param = {
        hp["parameter"]: hp["best_value"] for hp in model_parameters["parametric_space"]
    }

    space = [
        eval_parametric_space_dimension(d) for d in model_config["parametric_space"]
    ]

    x0 = list(hyper_param.values())

    @use_named_args(space)
    def train_function(**params):

        estimator = set_regressor(model_config["model"], **params)

        cv_mae = cross_validate_score(
            X_train,
            y_train,
            estimator,
            scoring=mean_absolute_error,
            n_folds=model_config["opt_config"]["n_folds"],
            fit_params=model_config["fit_parameters"],
        )

        print(f"\n\n{80 * '#'}")
        print("params:", list(params.values()))
        print("mae:", cv_mae)
        print(f"{50 * '-'}")

        return cv_mae

    res_gp = gp_minimize(
        train_function,
        space,
        x0=x0,
        y0=None,
        random_state=model_config["opt_config"]["random_state"],
        verbose=model_config["opt_config"]["verbose"],
        n_calls=model_config["opt_config"]["n_calls"],
        n_random_starts=model_config["opt_config"]["n_random_starts"],
    )

    model_config["fit_parameters"]["eval_set"] = None

    # for i, param in enumerate(model_config["parametric_space"]):
    #     param.update({"optimal_value": res_gp.x[i]})

    # return {
    #     "best_parameters": dict(zip(hyper_param.keys(), res_gp.x)),
    #     "best_score": res_gp.fun,
    # }

    return res_gp