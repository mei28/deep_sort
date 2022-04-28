import cv2
import pickle
import csv
from pdb import set_trace as pst
from typing import Union, Optional
from pathlib import Path
from tqdm import tqdm
import argparse

# colors
GRAY = [200, 200, 200]
WHITE = [255, 255, 255]
RED = [244, 67, 54][::-1]
BLUE = [33, 150, 243][::-1]
GREEN = [139, 195, 74][::-1]
ORANGE = [255, 87, 34][::-1]

# font
FONT_SIZE = 12
FONT_SCALE = 0.5


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)

    args = parser.parse_args()
    return args

def min_max_shape(im0,x:float,y:float)->tuple:
    """画像内に入るように抑え込む

    Args:
        im0: 画像
        x (float): 
        y (float): 

    Returns:
        x,y
    """
    
    max_height,max_width,_ = im0.shape
    x = max(0,x)
    y = max(0,y)
    x = min(x,max_width)
    y = min(y,max_height)

    return x,y


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
        overlay, (int(x1), int(y1)), (int(x2), int(y2)), color=kwargs["color"], thickness=2
    )
    im0 = cv2.addWeighted(overlay,kwargs['alpha'],im0,1-kwargs['alpha'],0)
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
    im0 = cv2.addWeighted(overlay, kwargs['alpha'],im0,1-kwargs['alpha'],0)

    x1,y1 = min_max_shape(im0,x1,y1)
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

    config={'color':RED,'alpha':0.4}

    for one_bbox in bboxes:
        frame_id, player_id = int(one_bbox[0]), int(one_bbox[1])
        assert frame_id == target_frame_id
        bbox =  one_bbox[2:6]
        bbox = list(map(lambda x: max(float(x),0),bbox))
        bbox = tlwh2tldr(bbox)
        im0 = draw_bbox(im0, bbox, **config)
        im0 = draw_id(im0, bbox, player_id, **config)
    return im0


def show_image(im0):
    cv2.imshow("img", im0)
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

        if _frame_id > _frame_id:
            break
    return ret


def get_changed_trackret(name: str):
    with open(
        f"/home/mei/Documents/deep_sort/exp/20220314/{name}_trackret.pickle", "rb"
    ) as f:
        changed_trackret = pickle.load(f)

    player_id, changed_frame_id = (
        changed_trackret["player_id"],
        changed_trackret["changed_frame_id"],
    )
    player_id.sort()
    changed_frame_id.sort()
    return player_id, changed_frame_id


def save_frame_img(im0, video_name: str, frame_id: int):
    """画像を保存する"""
    dst_path = Path(
        f"/home/mei/Documents/deep_sort/exp/20220314/create_img/{video_name}"
    )
    if not dst_path.exists():
        dst_path.mkdir(parents=True)
    cv2.imwrite(str(dst_path / f"{frame_id:0>8}.jpg"), im0)


if __name__ == "__main__":
    args = get_args()
    img_path: Path = Path(
        f"/home/mei/Documents/deep_sort/exp/20220310/data/{args.name}/img1"
    )
    detect_path: Path = Path(
        f"/home/mei/Documents/deep_sort/exp/20220310/data/{args.name}/{args.name}_deepsort_output.txt"
    )

    player_id, changed_frame_id = get_changed_trackret(args.name)
    bbox_list: list = load_bbox_data(str(detect_path))

    progress_bar = tqdm(changed_frame_id,total=len(changed_frame_id))
    for i,frame_id in enumerate(progress_bar):
        im0 = draw_one_frame(bbox_list, frame_id, img_path)
        # show_image(im0)
        save_frame_img(im0, args.name, frame_id)


