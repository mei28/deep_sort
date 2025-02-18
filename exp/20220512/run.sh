#!/usr/bin/bash

video_name="p008"
dst_path="/home/mei/Documents/deep_sort/exp/20220512/output/${video_name}/img"
img_path="/home/mei/Documents/deep_sort/exp/20220512/data/${video_name}/img1"
tracklet_path="/home/mei/Documents/deep_sort/exp/20220512/data/${video_name}_annotated_tracklet.csv"
bbox_path="/home/mei/Documents/deep_sort/exp/20220512/data/${video_name}/${video_name}_deepsort_output.txt"

pose_path="/home/mei/Documents/deep_sort/exp/20220512/data/detection_poses_${video_name}.pkl"
max_frame=109275

python exp/20220512/main.py \
  --dst_path $dst_path \
  --img_path $img_path \
  --tracklet_path $tracklet_path \
  --bbox_path $bbox_path \
  --max_frame $max_frame \
  --pose_path $pose_path
