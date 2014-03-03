#!/bin/bash

 ./crohme.py parsing -i ../TrainINKML_v3/ -f folds.dump -e 0 -s 12 -o ../parsing_0 > parsing_0.log

 ./crohme.py parsing -i ../TrainINKML_v3/ -f folds.dump -e 1 -s 02 -o ../parsing_1 > parsing_1.log

 ./crohme.py parsing -i ../TrainINKML_v3/ -f folds.dump -e 2 -s 01 -o ../parsing_2 > parsing_2.log

# ./crohme.py segment -i ../TrainINKML_v3/ -s all -o ../seg_all > segment_all.log
