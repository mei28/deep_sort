#! /usr/bin/bash

array=('p008' 'p029' 'p036' 'p037' 'p042')

for e in ${array[@]};do
  echo $e
  python exp/20220314/visualizer.py --name $e
done
