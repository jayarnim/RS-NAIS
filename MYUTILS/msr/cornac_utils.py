# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
from tqdm import tqdm  # Progress Bar 생성을 위해 추가
from ..config.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_PREDICTION_COL,
)


def predict(
    model,
    data,
    usercol=DEFAULT_USER_COL,
    itemcol=DEFAULT_ITEM_COL,
    predcol=DEFAULT_PREDICTION_COL,
):
    """Computes predictions of a recommender model from Cornac on the data.
    Can be used for computing rating metrics like RMSE.

    Args:
        model (cornac.models.Recommender): A recommender model from Cornac
        data (pandas.DataFrame): The data on which to predict
        usercol (str): Name of the user column
        itemcol (str): Name of the item column

    Returns:
        pandas.DataFrame: Dataframe with usercol, itemcol, predcol
    """
    uid_map = model.train_set.uid_map
    iid_map = model.train_set.iid_map
    predictions = [
        [
            getattr(row, usercol),
            getattr(row, itemcol),
            model.rate(
                user_idx=uid_map.get(getattr(row, usercol), len(uid_map)),
                item_idx=iid_map.get(getattr(row, itemcol), len(iid_map)),
            ),
        ]
        for row in data.itertuples()
    ]
    predictions = pd.DataFrame(data=predictions, columns=[usercol, itemcol, predcol])
    return predictions


def predict_ranking(
    model,
    data,
    usercol=DEFAULT_USER_COL,
    itemcol=DEFAULT_ITEM_COL,
    predcol=DEFAULT_PREDICTION_COL,
    remove_seen=False,
):
    """Computes predictions of recommender model from Cornac on all users and items in data.
    It can be used for computing ranking metrics like NDCG.

    Args:
        model (cornac.models.Recommender): A recommender model from Cornac
        data (pandas.DataFrame): The data from which to get the users and items
        usercol (str): Name of the user column
        itemcol (str): Name of the item column
        remove_seen (bool): Flag to remove (user, item) pairs seen in the training data

    Returns:
        pandas.DataFrame: Dataframe with usercol, itemcol, predcol
    """
    users, items, preds = [], [], []
    item = list(model.train_set.iid_map.keys())
    for uid, user_idx in tqdm(model.train_set.uid_map.items()):  # Progress Bar 생성을 위해 수정
        user = [uid] * len(item)
        users.extend(user)
        items.extend(item)
        score = model.score(user_idx).tolist()
        if model.name == 'EASEᴿ':  # Fix bug
            score = score[0]
        preds.extend(score)
    
    all_predictions = pd.DataFrame(
        data={usercol: users, itemcol: items, predcol: preds}
    )

    if remove_seen:
        tempdf = pd.concat(
            [
                data[[usercol, itemcol]],
                pd.DataFrame(
                    data=np.ones(data.shape[0]), columns=["dummycol"], index=data.index
                ),
            ],
            axis=1,
        )
        merged = pd.merge(tempdf, all_predictions, on=[usercol, itemcol], how="outer")
        return merged[merged["dummycol"].isnull()].drop("dummycol", axis=1)
    else:
        return all_predictions
