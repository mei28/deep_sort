# !/usr/bin/bash


annotator='A'
python exp/20220428/main.py \
        --data_path "exp/20220428/data/project-3-at-2022-04-08-02-02-ff297c55.csv"\
        --dst_path "exp/20220428/data/label_img/${annotator}" \
        --img_path "/home/mei/Documents/deep_sort/exp/20220428/data/img" \
        --max_frame 3698 \


annotator='B'
python exp/20220428/main.py \
        --data_path "exp/20220428/data/project-6-at-2022-04-08-02-03-f42996b2.csv"\
        --dst_path "exp/20220428/data/label_img/${annotator}" \
        --img_path "/home/mei/Documents/deep_sort/exp/20220428/data/img" \
        --max_frame 3698 \
