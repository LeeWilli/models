#!/bin/bash

SPLIT=validation  # or test
MODEL=ssd_inception_v2_coco_2017_11_17
ECORD_FILES=$(ls -1 ${SPLIT}_tfrecords/* | tr '\n' ',')
PYTHONPATH=$PYTHONPATH:$(readlink -f ..) \
python -m object_detection.inference.infer_detections \
  --input_tfrecord_paths=$TF_RECORD_FILES \
  --output_tfrecord_path=${MODEL}_${SPLIT}_detections.tfrecord-00000-of-00001 \
  --inference_graph=${MODEL}/frozen_inference_graph.pb \
  --discard_image_pixels

NUM_SHARDS=1  # Set to NUM_GPUS if using the parallel evaluation script above
mkdir -p ${MODEL}_${SPLIT}_eval_metrics
echo "
label_map_path: '../object_detection/data/mscoco_label_map.pbtxt'
tf_record_input_reader: { input_path: '${MODEL}_${SPLIT}_detections.tfrecord@${NUM_SHARDS}'  }
" > ${MODEL}_${SPLIT}_eval_metrics/${SPLIT}_input_config.pbtxt
echo "
metrics_set: 'open_images_metrics'
" > ${MODEL}_${SPLIT}_eval_metrics/${SPLIT}_eval_config.pbtxt


python -m  object_detection.metrics.offline_eval_map_corloc \
  --eval_dir=${MODEL}_${SPLIT}_eval_metrics \
  --eval_config_path=${MODEL}_${SPLIT}_eval_metrics/${SPLIT}_eval_config.pbtxt \
  --input_config_path=${MODEL}_${SPLIT}_eval_metrics/${SPLIT}_input_config.pbtxt


