import cv2
import pickle
import csv
from pdb import set_trace as pst
from ipdb import set_trace as ist
from typing import Union, Optional, List
from pathlib import Path
from tqdm import tqdm
import argparse
from dataclasses import dataclass, field
from collections import OrderedDict

# colors
GRAY = [200, 200, 200]
WHITE = [255, 255, 255]
RED = [244, 67, 54][::-1]
BLUE = [33, 150, 243][::-1]
GREEN = [139, 195, 74][::-1]
ORANGE = [255, 87, 34][::-1]
YELLOW = [255, 193, 7][::-1]

# font
FONT_SIZE = 12
FONT_SCALE = 0.5
LABEL_FONT_SCALE = 2.0

colors = {
    "GRAY": GRAY,
    "WHITE": WHITE,
    "RED": RED,
    "BLUE": BLUE,
    "GREEN": GREEN,
    "ORANGE": ORANGE,
    "YELLOW": YELLOW,
}
# segments for plotting
segments = OrderedDict(
    {
        1: [5, 6],
        2: [5, 11],
        3: [11, 12],
        4: [12, 6],
        5: [5, 7],
        6: [7, 9],
        7: [6, 8],
        8: [8, 10],
        9: [11, 13],
        10: [13, 15],
        11: [12, 14],
        12: [14, 16],
    }
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)

    args = parser.parse_args()
    return args


def min_max_shape(im0, x: float, y: float) -> tuple:
    """画像内に入るように抑え込む

    Args:
        im0: 画像
        x (float):
        y (float):

    Returns:
        x,y
    """

    max_height, max_width, _ = im0.shape
    x = max(0, x)
    y = max(0, y)
    x = min(x, max_width)
    y = min(y, max_height)

    return x, y


def get_label_color(label: str):
    """ラベルに応じた色を返す"""
    if label == "OA":
        return RED
    elif label == "OB":
        return BLUE
    elif label == "XA":
        return YELLOW
    elif label == "XB":
        return GREEN
    else:
        raise ValueError


def draw_labels(im0, label: str = None, **kwargs):
    """labelを動画に書き込む"""
    if label is None:
        return im0.copy()
    color = get_label_color(label)
    cv2.putText(
        img=im0,
        text=f"{label}",
        org=(0, 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=LABEL_FONT_SCALE,
        color=color,
        thickness=2,
    )

    return im0.copy()


def draw_pose(im0, _pose, **kwargs):
    """姿勢を書き込む"""
    pose = _pose[2]

    for _, seg in segments.items():
        pt1 = (int(pose[seg[0], 0]), int(pose[seg[0], 1]))
        pt2 = (int(pose[seg[1], 0]), int(pose[seg[1], 1]))
        cv2.line(im0, pt1, pt2, **kwargs)
    return im0.copy()


def draw_bbox(im0, bbox: Union[tuple, list] = [], **kwargs):
    """選手矩形を描画する
    Args:
        im0 :画像
        bbox tuple: tldr形式の座標
    """
    x1, y1, x2, y2 = bbox
    # x1,y1 = min_max_shape(im0,x1,y1)
    # x2,y2 = min_max_shape(im0,x2,y2)
    overlay = im0.copy()
    cv2.rectangle(
        overlay,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color=kwargs["color"],
        thickness=2,
    )
    im0 = cv2.addWeighted(overlay, kwargs["alpha"], im0, 1 - kwargs["alpha"], 0)
    return im0.copy()


def draw_id(im0, bbox: Union[tuple, list], id_num: int, **kwargs):
    """その矩形領域に割り当てられている数字を描画する
    Args:
        im0: 画像 tldr形式にする
        bbox: 矩形座標
    """
    num_height = FONT_SIZE
    num_width = FONT_SIZE * (len(str(id_num)) + 2)
    x1, y1, _, _ = bbox
    overlay = im0.copy()
    cv2.rectangle(
        overlay,
        (int(x1), int(y1)),
        (int(x1) + num_width, int(y1) + num_height),
        color=kwargs["color"],
        thickness=-1,
    )
    im0 = cv2.addWeighted(overlay, kwargs["alpha"], im0, 1 - kwargs["alpha"], 0)

    x1, y1 = min_max_shape(im0, x1, y1)
    cv2.putText(
        img=im0,
        text=f"{id_num}",
        org=(int(x1 + FONT_SIZE), int(y1 + FONT_SIZE + 0.5)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=FONT_SCALE,
        color=WHITE,
        thickness=2,
    )

    return im0.copy()


def tlwh2tldr(bbox: Union[tuple, list]):
    """tlwh形式からtldr形式を返す。
    Args:
        bbox: tlwh形式の座標系
    Returns:
        tldr形式の座標系
    """
    t, l, w, h = list(map(float, bbox))
    x1 = l
    y1 = t
    x2 = l + h
    y2 = t + w
    # return [x1,y1,x2,y2]
    return [y1, x1, y2, x2]


def load_bbox_data(path_name: str) -> list:
    """bboxの情報を獲得する
    Args:
        path_name (str):パスの名前
    Returns:
        リストにしたbbox情報(MOT形式)
    """
    bbox_list: list = []
    with open(str(path_name)) as f:
        reader = csv.reader(f)
        for row in reader:
            bbox_list.append(row)

    return bbox_list


def load_img(base_path: Path, target_frame_id: int):
    """画像を返す
    Args:
        base_path: 対象画像のフォルダ
        target_frame_id: ほしい画像のフレーム
    Returns:
        im0
    """
    img_path = base_path / f"{target_frame_id:0>8}.jpg"
    if not img_path.exists():
        raise ValueError

    im0 = cv2.imread(str(img_path))
    return im0


def draw_one_frame(_bboxes: list, target_frame_id: int, img_path: Path):
    """
    指定したフレームの矩形情報を描画する
    Args:
        _bboxes (list): 矩形情報たち
        target_frame_id (int): 指定するフレーム番号
        img_path (Path): 描画するのに使う画像
    """
    im0 = load_img(img_path, target_frame_id)
    bboxes: list = get_masked_bboxes(_bboxes, target_frame_id)

    config = {"color": RED, "alpha": 0.4}

    for one_bbox in bboxes:
        frame_id, player_id = int(one_bbox[0]), int(one_bbox[1])
        assert frame_id == target_frame_id
        bbox = one_bbox[2:6]
        bbox = list(map(lambda x: max(float(x), 0), bbox))
        bbox = tlwh2tldr(bbox)
        im0 = draw_bbox(im0, bbox, **config)
        im0 = draw_id(im0, bbox, player_id, **config)
    return im0


def show_image(im0, frame_id: int = 0):
    cv2.imshow(f"img{frame_id}", im0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_masked_bboxes(bboxes: list, target_frame_id: int) -> list:
    """
    bboxesの中で対象となるフレームだけ抽出する
    Args:
        bboxes (list):矩形情報
        target_frame_id (int): 対象のフレーム
    Returns:
        マスクされた矩形情報
    """
    ret: list = []
    for _bbox in bboxes:
        _frame_id = int(_bbox[0])
        if _frame_id == target_frame_id:
            ret.append(_bbox)

        if _frame_id > target_frame_id:
            break
    return ret


def get_masked_poses(poses: list, target_frame_id: int) -> list:
    ret: list = []
    for pose in poses:
        _frame_id = pose[0]
        if _frame_id == target_frame_id:
            ret.append(pose)
        if _frame_id > target_frame_id:
            break
    return ret


def get_changed_trackret(name: str):
    with open(
        f"/home/mei/Documents/deep_sort/exp/20220428/{name}_trackret.pickle", "rb"
    ) as f:
        changed_trackret = pickle.load(f)

    player_id, changed_frame_id = (
        changed_trackret["player_id"],
        changed_trackret["changed_frame_id"],
    )
    player_id.sort()
    changed_frame_id.sort()
    return player_id, changed_frame_id


def save_frame_img(im0, dst_path: Path, frame_id: int):
    """画像を保存する"""

    if not dst_path.exists():
        dst_path.mkdir(parents=True)
    cv2.imwrite(str(dst_path / f"{frame_id:0>8}.jpg"), im0)


def get_player_config(
    player_id: Union[int, str], tracklets: tuple
) -> Union[dict, bool]:
    """プレイヤーの描画コンフィグを獲得する
    Args:
        player_id: ターゲットのプレイヤーid
        tracklets: 各トラックレット
    Returns:
        Union[dict,False]: 存在する場合はコンフィグ、存在しない場合はFalse
    """
    player1_tracklet, player2_tracklet = tracklets
    player1_tracklet, player2_tracklet = set(player1_tracklet), set(player2_tracklet)
    player1_tracklet.remove(-1)
    player2_tracklet.remove(-1)

    if player_id in player1_tracklet:
        return {"color": colors["RED"], "alpha": 0.4}
    elif player_id in player2_tracklet:
        return {"color": colors["BLUE"], "alpha": 0.4}
    else:
        return False
