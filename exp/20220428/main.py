import sys

sys.path.append("./exp/20220428/utils")

import argparse

from utils.load_data import load_data, list2df
from utils.visualize import load_img, draw_labels, save_frame_img
from utils.frame import sec2frame
import tqdm
from pdb import set_trace as pst
from pathlib import Path


def main():
    data_path = Path("exp/20220428/data/project-3-at-2022-04-08-02-02-ff297c55.csv")
    dst_path = Path("exp/20220428/data/label_img")
    img_path = Path("/home/mei/Documents/deep_sort/exp/20220428/data/img")
    MAX_FRAME = 3698

    # annotated data
    data1 = load_data(str(data_path))
    data1 = list2df(data1)

    # annotated frames
    frames = (data1["start"].apply(lambda x: sec2frame(x))).tolist()
    frames_set = set(frames)

    label = None
    for i_frame in range(1, MAX_FRAME):
        im0 = load_img(img_path, i_frame)
        if i_frame in frames_set:
            idx = frames.index(i_frame)
            label = data1.loc[idx, "labels"]
        im0 = draw_labels(im0, label)
        # show_image(im0,i)
        save_frame_img(im0, dst_path, i_frame)

    pst()
    pass


def convert(s):
    if "A" in s:
        return s.replace("A", "B")
    else:
        return s.replace("B", "A")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dst_path", type=str, required=True)
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--max_frame", type=int, required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    data_path: Path = Path(args.data_path)
    MAX_FRAME: int = args.max_frame
    img_path: Path = Path(args.img_path)
    dst_path: Path = Path(args.dst_path)

    # annotated data
    data1 = load_data(str(data_path))
    data1 = list2df(data1)

    if str(data_path) == "exp/20220428/data/project-6-at-2022-04-08-02-03-f42996b2.csv":
        data1["labels"] = data1["labels"].apply(lambda x: convert(x))

    # annotated frames
    frames = (data1["start"].apply(lambda x: sec2frame(x))).tolist()
    frames_set = set(frames)

    label = None
    progress = tqdm.tqdm(range(1, MAX_FRAME))
    for i_frame in progress:
        im0 = load_img(img_path, i_frame)
        if i_frame in frames_set:
            idx = frames.index(i_frame)
            label = data1.loc[idx, "labels"]
        im0 = draw_labels(im0, label)
        # show_image(im0,i)
        save_frame_img(im0, dst_path, i_frame)

    pass
