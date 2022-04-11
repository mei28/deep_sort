# !/usr/bin/bash

# python runs/run_pingpong.py --sequence_dir data/pingpong --detection_file data/pingpong/det/detection.npy --output_file data/out

target_array=(p008 p029 p036 p037 p042)

for file in "${target_array[@]}"; do
  echo $file
  # echo exp/20220310/data/$file 
  # echo exp/20220310/data/$file/det/detection_$file.npy 
  # echo exp/20220310/data/$file/${file}_deepsort_output 

  python exp/20220310/run_pingpong.py --sequence_dir exp/20220310/data/$file --detection_file exp/20220310/data/$file/det/detection_$file.npy --output_file exp/20220310/data/${file}/${file}_deepsort_output.txt 
done
