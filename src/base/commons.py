import re
import git
import pandas as pd
import numpy as np
import json
import yaml
import dill as pickle
from sklearn.base import TransformerMixin


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def get_last_git_tag() -> str:
    """
    Get the latest git tag.

    Returns
    -------
    str
        Latest git tag
    """

    repo = git.Repo()

    latest_tag = None

    try:
        latest_tag = sorted(repo.tags, key=lambda t: t.commit.committed_datetime)[
            -1
        ].name

    except IndexError:
        raise IndexError(
            "No git tags found. You can add one through `git tag <tag_name>`"
        )

    return latest_tag


def to_snake_case(string: str) -> str:
    """Converts a string to snake case.

    Parameters
    ----------
    string : str
        Any input string

    Returns
    -------
    str
        The string converted to snake case format
    """
    string = string.strip().replace(" ", "_")

    string = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", string)

    string = re.sub("([a-z0-9])([A-Z])", r"\1_\2", string).lower()

    while "__" in string:
        string = string.replace("__", "_")

    return string


def dataframe_transformer(
    dataframe: pd.DataFrame, transformer: TransformerMixin
) -> pd.DataFrame:
    """
    Applies a sklearn transformation to the input dataframe and converts the
    resulting array in a dataframe, with the same column names. The transformations
    where the column numbers is changed, as PolinomialFeatures and PCA for example are
    not suported by this method.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe
    transformer : TransformerMixin
        Scikit-Learn-like transformation to be applied

    Returns
    -------
    pd.DataFrame
        The transformed dataframe
    """
    transformed_array = transformer.transform(dataframe)

    if transformed_array.shape[1] == len(dataframe.columns):
        result = pd.DataFrame(
            transformed_array,
            index=dataframe.index,
            columns=dataframe.columns,
        )

    else:
        raise ValueError(
            """The transformation do not preserve the number \
        of columns. So, the transformed data cannot be converted to a dataframe \
        with same column names.
        """
        )

    return result


def dump_json(obj, filepath, *args, **kwargs):

    with open(filepath, "w") as file:
        json.dump(obj, file, cls=NpEncoder, *args, **kwargs)


def load_json(filepath, *args, **kwargs):

    with open(filepath, "r") as file:
        data = json.load(file, *args, **kwargs)

    return data


def dump_yaml(obj, filepath, *args, **kwargs):

    with open(filepath, "w") as file:
        yaml.dump(obj, file, *args, **kwargs)


def load_yaml(filename, *args, **kwargs):

    with open(filename, "r") as file:
        data = yaml.safe_load(file, *args, **kwargs)

    return data


def dump_pickle(obj, filepath, *args, **kwargs):

    with open(filepath, "wb") as file:
        pickle.dump(obj, file, *args, **kwargs)


def load_pickle(filename, *args, **kwargs):

    with open(filename, "rb") as file:
        data = pickle.load(file, *args, **kwargs)

    return data