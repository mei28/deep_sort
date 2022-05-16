## フレームが切り替わったときのプレーヤのidとフレームidを取得する
## トラックレットに用いる

import pickle
import os
import sys
from pathlib import Path
from pdb import set_trace as pst
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

if __name__ == "__main__":

    names: list = ["p008", "p029", "p036", "p037", "p042"]
    taget_file_path: list = [
        f"exp/20220310/data/{name}/{name}_deepsort_output.txt" for name in names
    ]

    for i, name in zip(taget_file_path, names):
        # load pickle file
        df: pd.DataFrame = pd.read_csv(i, header=None)

        player_id_set: set = set()
        player_changed_frame_id: set = set()

        output_df = df.groupby(0)[1].apply(list).reset_index()
        output_df = output_df.rename(columns={0: "frmae", 1: "player_id"})
        output_df = output_df[output_df["player_id"] != output_df["player_id"].shift(1)]
        output_df.to_csv(f"exp/20220428/{name}_frame_player.csv", index=False)

        # check changed id
        # data: np.ndarray = df.iloc[:, [0, 1]].to_numpy()
        # progress_bar = tqdm(data, total=len(data))

        # for frame_id, player_id in progress_bar:
        #     if player_id not in player_id_set:
        #         player_id_set.add(player_id)
        #         player_changed_frame_id.add(frame_id)

        # ret_dct: dict = {
        #     "player_id": list(player_id_set),
        #     "changed_frame_id": list(player_changed_frame_id),
        # }
        # pst()

        # with open(f"exp/20220428/{name}_trackret.pickle", "wb") as f:
        #     pickle.dump(ret_dct, f)
    pass
