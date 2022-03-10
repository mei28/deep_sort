#!/usr/bin/bash


array=(`ls . | grep -e "mp4"`)

for file in "${array[@]}"; do
  file_name=`basename $file .mp4`

  mkdir -p $file_name

  # move movie to file_name
  mv $file_name.mp4 $file_name

  # move detection file_name
  mkdir -p $file_name/det
  mv detection_$file_name.npy $file_name/det

done
