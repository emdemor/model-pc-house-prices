import re
import git
import pandas as pd


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


def to_snake_case(name):

    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)

    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()

    name = name.replace("__", "_")

    return name


def dataframe_transformer(dataframe, transformer):
    return pd.DataFrame(
        transformer.transform(dataframe),
        index=dataframe.index,
        columns=dataframe.columns,
    )
