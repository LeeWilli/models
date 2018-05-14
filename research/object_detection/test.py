# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Training executable for detection models.

This executable is used to train DetectionModels. There are two ways of
configuring the training job:

1) A single pipeline_pb2.TrainEvalPipelineConfig configuration file
can be specified by --pipeline_config_path.

Example usage:
    ./train \
        --logtostderr \
        --train_dir=path/to/train_dir \
        --pipeline_config_path=pipeline_config.pbtxt

2) Three configuration files can be provided: a model_pb2.DetectionModel
configuration file to define what type of DetectionModel is being trained, an
input_reader_pb2.InputReader file to specify what training data will be used and
a train_pb2.TrainConfig file to configure training parameters.

Example usage:
    ./train \
        --logtostderr \
        --train_dir=path/to/train_dir \
        --model_config_path=model_config.pbtxt \
        --train_config_path=train_config.pbtxt \
        --input_config_path=train_input_config.pbtxt
"""

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

import logging

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from utils import label_map_util
from utils import visualization_utils as vis_util
from utils import dataset_util


tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('input_path', '', 'input path')
flags.DEFINE_string('output_path', '', 'output path')
flags.DEFINE_string('example_path', '', 'example path')
flags.DEFINE_string('path_to_labels', '', 'labels path')
flags.DEFINE_string('inference_path', None,
                       'Path to the inference graph with embedded weights.')
FLAGS = flags.FLAGS

def convert(filename_in, filename_out, ffmpeg_executable="ffmpeg"):
    import subprocess
    command = [ffmpeg_executable, "-i", filename_in, "-c:v", "libx264",
               "-preset", "slow", "-crf", "21", filename_out]
    subprocess.call(command)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def main(_):
    assert FLAGS.inference_path, '`inference_path` is missing.'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = os.path.join(FLAGS.inference_path, 'frozen_inference_graph.pb')
    print("DEBUG:"+ PATH_TO_CKPT)
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = FLAGS.path_to_labels

    NUM_CLASSES = 3

    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    # Loading label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Size, in inches, of the output images.
    IMAGE_SIZE = (12, 8)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            examples_list = dataset_util.read_examples_list(FLAGS.example_path)
            print("Debug: " + FLAGS.example_path)
            print("Debug: " + str(len(examples_list)))
            for idx, example in enumerate(examples_list):
              if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(examples_list))
              image_path = os.path.join(FLAGS.input_path, example)
              try:
                image = Image.open(image_path)
              except FileNotFoundError:
                print(image_path)
                continue
              # the array based representation of the image will be used later in order to prepare the
              # result image with boxes and labels on it.
              image_np = load_image_into_numpy_array(image)
              # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
              image_np_expanded = np.expand_dims(image_np, axis=0)
              # Actual detection.
              (boxes, scores, classes, num) = sess.run(
                  [detection_boxes, detection_scores, detection_classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})
              # Visualization of the results of a detection.
              vis_util.visualize_boxes_and_labels_on_image_array(
                  image_np,
                  np.squeeze(boxes),
                  np.squeeze(classes).astype(np.int32),
                  np.squeeze(scores),
                  category_index,
                  use_normalized_coordinates=True,
                  line_thickness=8)
              result = Image.fromarray(image_np)
              result.save(os.path.join(FLAGS.output_path, example))

if __name__ == '__main__':
  tf.app.run()
