import pandas as pd
from pathlib import Path
from typing import Union, Any, List, Tuple
import pickle
import csv
import json


def load_data(path: str) -> Any:
    df = pd.read_csv(path)
    js = json.loads(df["tricks"][0])
    return js


def list2df(data) -> pd.DataFrame:
    output_df = pd.DataFrame()
    for d in data:
        _df = pd.DataFrame.from_dict(d)
        output_df = pd.concat([output_df, _df], axis=0).reset_index(drop=True)
    return output_df


def load_annotated_trackret_file(path: Union[str, Path]) -> pd.DataFrame:
    """trackletのdfを返す
    Args:
        path (str,Path): ぱす
    Returns:
        pd.DataFrame: [frame,player1,player2]
    """
    if isinstance(path, Path):
        path = str(path)
    ret_df = pd.read_csv(path)
    ret_df = ret_df.replace("-", -1)

    ret_df = ret_df.dropna(subset=["frame"])
    ret_df = ret_df.astype({k: int for k in ret_df.columns})

    return ret_df


def get_player_id_dict(df: pd.DataFrame):
    """フレーム番号、player{1,2}のdictを返す
    Returns:
        dict[frame : {player1,player2}]
    """
    ret_dct = {
        f: {"player1": p1, "player2": p2}
        for f, p1, p2 in zip(
            df["frame"].to_numpy(), df["player1"].to_numpy(), df["player2"].to_numpy()
        )
    }

    return ret_dct


def get_player_id_list(df: pd.DataFrame) -> Tuple[list, list]:
    """フレーム番号、player{1,2}のlistを返す
    Returns:
        list, list
    """
    player1: List[int] = list(set(df["player1"].astype(int).to_numpy().tolist()))
    player2: List[int] = list(set(df["player2"].astype(int).to_numpy().tolist()))

    return player1, player2


def load_bbox_data(path_name: Union[str, Path]) -> list:
    """bboxの情報を獲得する
    Args:
        path_name (str):パスの名前
    Returns:
        リストにしたbbox情報(MOT形式)
    """
    bbox_list: list = []
    with open(str(path_name)) as f:
        try:
            reader = csv.reader(f)
            for row in reader:
                bbox_list.append(row)
        except IOError:
            raise ValueError

    return bbox_list


def load_pose_data(path_name: Union[str, Path]) -> list:
    with open(str(path_name), "rb") as f:
        try:
            data: list = pickle.load(f)
            return data
        except IOError:
            raise ValueError


if __name__ == "__main__":
    pass
