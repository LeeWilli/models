#!/bin/bash

wget -c http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz>log.txt>&1 &
wget -c http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz>log.txt>&1 &

