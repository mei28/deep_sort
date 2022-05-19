import sys

sys.path.append("./exp/20220428/utils")

import argparse

from utils.load_data import (
    load_data,
    list2df,
    load_annotated_trackret_file,
    get_player_id_list,
    get_player_id_dict,
    load_bbox_data,
    load_pose_data,
)
from utils.visualize import (
    load_img,
    draw_labels,
    save_frame_img,
    show_image,
    get_masked_bboxes,
    get_masked_poses,
    tlwh2tldr,
    draw_id,
    draw_bbox,
    colors,
    draw_pose,
)
from utils.frame import sec2frame
import tqdm
from pdb import set_trace as pst
from ipdb import set_trace as ist
import pandas as pd
from pathlib import Path
from typing import Union, Tuple


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst_path", type=str, required=True)
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--max_frame", type=int, required=True)
    parser.add_argument("--bbox_path", type=str, required=True)
    parser.add_argument("--tracklet_path", type=str, required=True)
    parser.add_argument("--pose_path", type=str, required=True)

    args = parser.parse_args()
    return args


def get_player_config(
    player_id: Union[int, str], tracklets: tuple
) -> Union[dict, bool]:
    player1_tracklet, player2_tracklet = tracklets
    player1_tracklet, player2_tracklet = set(player1_tracklet), set(player2_tracklet)

    if player_id in player1_tracklet:
        return {"color": colors["RED"], "alpha": 0.4}
    elif player_id in player2_tracklet:
        return {"color": colors["BLUE"], "alpha": 0.4}
    else:
        return False


def draw_one_frame(
    _bboxes: list, target_frame_id: int, img_path: Path, tracklets: tuple, _poses: list
):
    """
    指定したフレームの矩形情報を描画する
    Args:
        _bboxes (list): 矩形情報たち
        target_frame_id (int): 指定するフレーム番号
        img_path (Path): 描画するのに使う画像
    """
    im0 = load_img(img_path, target_frame_id)
    bboxes: list = get_masked_bboxes(_bboxes, target_frame_id)
    poses: list = get_masked_poses(_poses, target_frame_id)

    for one_bbox in bboxes:
        frame_id, player_id = int(one_bbox[0]), int(one_bbox[1])
        assert frame_id == target_frame_id
        config = get_player_config(player_id, tracklets)
        if not config:
            config = {"color": colors["GRAY"], "alpha": 0.4}
        bbox = one_bbox[2:6]
        bbox = list(map(lambda x: max(float(x), 0), bbox))
        bbox = tlwh2tldr(bbox)
        im0 = draw_bbox(im0, bbox, **config)
        im0 = draw_id(im0, bbox, player_id, **config)

    for pose in poses:
        config = {"color": colors["GREEN"]}
        im0 = draw_pose(im0, pose, **config)
    return im0


if __name__ == "__main__":

    args = get_args()

    MAX_FRAME: int = args.max_frame
    img_path: Path = Path(args.img_path)
    dst_path: Path = Path(args.dst_path)
    bbox_path: Path = Path(args.bbox_path)
    tracklet_path: Path = Path(args.tracklet_path)
    pose_path: Path = Path(args.pose_path)

    # tracklet data
    tracklet_df = load_annotated_trackret_file(tracklet_path)
    player1_tracklet, player2_tracklet = get_player_id_list(tracklet_df)

    # pose data
    pose_data = load_pose_data(pose_path)

    bbox_list: list = load_bbox_data(str(bbox_path))

    for i_frame in tqdm.tqdm(range(1, MAX_FRAME)):
        im0 = load_img(img_path, i_frame)
        im0 = draw_one_frame(
            bbox_list,
            i_frame,
            img_path,
            (player1_tracklet, player2_tracklet),
            pose_data,
        )
        # show_image(im0, i_frame)
        save_frame_img(im0, dst_path, frame_id=i_frame)
