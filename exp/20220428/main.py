import sys

sys.path.append("./exp/20220428/utils")

from utils.load_data import *
from utils.visualize import *
from utils.frame import *
from pdb import set_trace as pst
from pathlib import Path

if __name__ == "__main__":

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
