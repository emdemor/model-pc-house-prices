import xgboost
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold


def cross_validate_score(
    X,
    y,
    estimator,
    scoring,
    verbose=1,
    n_folds=5,
    random_state=42,
    fit_params={},
    minimize=True,
):

    scores = []

    data = X.assign(y=y)

    data["fold"] = generate_folds(data, n_folds=n_folds, random_state=random_state)

    iterator = (
        range(n_folds) if verbose < 1 else tqdm(range(n_folds), desc="Cross validation")
    )

    for fold in iterator:

        # Separando os dados de treinamento para essa fold
        train_data = data[data["fold"] != fold].copy()

        # Separando os dados de teste para esse fold
        test_data = data[data["fold"] == fold].copy()

        X_1 = train_data.drop(columns=["fold", "y"]).values

        X_2 = test_data.drop(columns=["fold", "y"]).values

        y_1 = train_data["y"].values

        y_2 = test_data["y"].values

        if estimator.__class__ in [
            xgboost.sklearn.XGBRegressor,
            xgboost.sklearn.XGBClassifier,
        ]:
            fit_params["eval_set"] = [(X_2, y_2)]

        try:
            estimator.fit(X_1, y_1, **fit_params)
        except:
            estimator.fit(X_1, y_1, **fit_params)

        scores.append(scoring(y_2, estimator.predict(X_2)))

    if minimize:
        avg_score = np.mean(scores)
    else:
        avg_score = -np.mean(scores)

    return avg_score


def generate_folds(train, n_folds=5, shuffle=True, random_state=42):

    temp = train.copy().reset_index(drop=True)

    # Instaciando o estritificador
    kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)

    # Gerando os index com os folds
    stratified_folds = list(kf.split(X=temp.drop(columns="y"), y=temp["y"]))

    for fold_index in range(n_folds):

        train_index, validation_index = stratified_folds[fold_index]

        temp.loc[temp[temp.index.isin(validation_index)].index, "fold"] = fold_index

    return temp["fold"].astype(int)


def generate_stratified_folds(train, n_folds=5, shuffle=True, random_state=42):

    temp = train.copy().reset_index(drop=True)

    # Instaciando o estritificador
    skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)

    # Gerando os index com os folds
    stratified_folds = list(skf.split(X=temp.drop(columns="y"), y=temp["y"]))

    for fold_index in range(n_folds):

        train_index, validation_index = stratified_folds[fold_index]

        temp.loc[temp[temp.index.isin(validation_index)].index, "fold"] = fold_index

    return temp["fold"].astype(int)