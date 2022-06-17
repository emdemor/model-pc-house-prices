from skopt.space import Real, Categorical, Integer


def eval_parametric_space_dimension(parameter_dict):

    if parameter_dict["type"] == "real":
        return Real(
            low=parameter_dict["range"][0],
            high=parameter_dict["range"][1],
            name=parameter_dict["parameter"],
        )

    elif parameter_dict["type"] == "integer":
        return Integer(
            low=parameter_dict["range"][0],
            high=parameter_dict["range"][1],
            name=parameter_dict["parameter"],
        )

    elif parameter_dict["type"] == "categorical":
        return Categorical(
            categories=parameter_dict["categories"],
            name=parameter_dict["parameter"],
        )

    else:
        return None
