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
"""Utility functions for detection inference."""
from __future__ import division

import tensorflow as tf
import numpy as np
import json


#sys.path.append('/home/wangli/code/projects/UVA/video-recognition-ae/yolo/darkflow/darkflow/')
from ..cython_utils.cy_yolo2_findboxes import box_constructor

from object_detection.core import standard_fields

import pdb
def build_input(meta, tfrecord_paths):
  """Builds the graph's input.

  Args:
    tfrecord_paths: List of paths to the input TFRecords

  Returns:
    serialized_example_tensor: The next serialized example. String scalar Tensor
    image_tensor: The decoded image of the example. Uint8 tensor,
        shape=[1, None, None,3]
  """
  filename_queue = tf.train.string_input_producer(
      tfrecord_paths, shuffle=False, num_epochs=1)

  tf_record_reader = tf.TFRecordReader()
  _, serialized_example_tensor = tf_record_reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example_tensor,
      features={
          standard_fields.TfExampleFields.image_encoded:
              tf.FixedLenFeature([], tf.string),
      })
  encoded_image = features[standard_fields.TfExampleFields.image_encoded]
  image_tensor = tf.image.decode_image(encoded_image, channels=3)
  image_tensor_float = tf.image.convert_image_dtype(image_tensor, tf.float32)
  image_tensor_float.set_shape([None, None, 3])
  h, w, c = meta['inp_size']
  image_tensor_float = tf.image.resize_images(image_tensor_float, [h,w])
  image_tensor_float = tf.expand_dims(image_tensor_float, 0)

  return serialized_example_tensor, image_tensor_float


def build_meta( meta_file):
    """
    return a meta
    """
    with open(meta_file, 'r') as fp:
        meta = json.load(fp)
    return meta

def build_inference_graph(image_tensor, inference_graph_path):
  """Loads the inference graph and connects it to the input image.

  Args:
    image_tensor: The input image. uint8 tensor, shape=[1, None, None, 3]
    inference_graph_path: Path to the inference graph with embedded weights

  Returns:
    detected_boxes_tensor: Detected boxes. Float tensor,
        shape=[num_detections, 4]
    detected_scores_tensor: Detected scores. Float tensor,
        shape=[num_detections]
    detected_labels_tensor: Detected labels. Int64 tensor,
        shape=[num_detections]
  """
  with tf.gfile.Open(inference_graph_path, 'rb') as graph_def_file:
    graph_content = graph_def_file.read()
  graph_def = tf.GraphDef()
  graph_def.MergeFromString(graph_content)

  tf.import_graph_def(
      graph_def, name='', input_map={'input': image_tensor})

  g = tf.get_default_graph()

  detected_boxes_tensor = tf.squeeze(
      g.get_tensor_by_name('output:0'), 0)

  return detected_boxes_tensor

def process_box(meta, b, h, w, threshold):
    max_indx = np.argmax(b.probs)
    max_prob = b.probs[max_indx]
    label = meta['labels'][max_indx]
    #pdb.set_trace()
    if max_prob > threshold:
        left  = int ((b.x - b.w/2.) * w)
        right = int ((b.x + b.w/2.) * w)
        top   = int ((b.y - b.h/2.) * h)
        bot   = int ((b.y + b.h/2.) * h)
        if left  < 0    :  left = 0
        if right > w - 1: right = w - 1
        if top   < 0    :   top = 0
        if bot   > h - 1:   bot = h - 1
        mess = '{}'.format(label)
        #return (left, right, top, bot, mess, max_indx, max_prob)
        return (top, left, bot, right, mess, max_indx, max_prob)
    return None

def yolo2googleout( meta, category_index, yolo_out, threshold=0.3):
    """convert out to desired format
    :param feed_dict:
    :param img_shape:
    :return: out
    :return: detection_out
    """
    out = yolo_out.copy()
    boxes = box_constructor(meta, out)
    h, w, c = meta['inp_size']
    out = [process_box(meta, b, h, w, threshold) for b in boxes]
    out = [out1 for out1 in out if out1] #remove empty boxes
    # out format: [[left, right, top, bot, mess, max_indx, confidence], ...]
    #pdb.set_trace()
    try:
        detection_out = np.array([norm_boxes(meta, out1[:4]) for out1 in out])
        #pdb.set_trace()
    except ValueError:
        #pdb.set_trace()
        detection_out = []
    scores = np.array([out1[-1] for out1 in out])
    messes = [out1[-3] for out1 in out]
    ## the id of objects in coco is different between google detection and yolo detection
    ## change the id of yolo to google expression through category_index
    classes = []
    for mess in messes:
        c = [category['id'] for key, category in category_index.items() if category['name'] == mess]
        if c:
            classes.append(c[0])
        else:
            classes.append(0)

    classes = np.array(classes)
    classes = classes.reshape(-1,)
    #classes = np.array([out1[-2] for out1 in out])
    return detection_out, scores, classes #messes

def to_tlbr(tlwh):
    """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
    `(top left, bottom right)`.
    """
    ret = np.array(tlwh)
    ret[2:] += ret[:2]
    return ret

def norm_boxes(meta, boxes):
    """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
    `(top left, bottom right)`.
    """
    h, w, c = meta['inp_size']
    ret = np.array(boxes).astype(float)
    ret[0] = ret[0]/h
    ret[2] = ret[2]/h
    ret[1] = ret[1]/w
    ret[3] = ret[3]/w
    #pdb.set_trace()
    return ret

def infer_detections_and_add_to_example(
    meta, category_index, serialized_example_tensor, detected_boxes_tensor, discard_image_pixels):
  """Runs the supplied tensors and adds the inferred detections to the example.

  Args:
    serialized_example_tensor: Serialized TF example. Scalar string tensor
    detected_boxes_tensor: Detected boxes. Float tensor,
        shape=[num_detections, 4]
    detected_scores_tensor: Detected scores. Float tensor,
        shape=[num_detections]
    detected_labels_tensor: Detected labels. Int64 tensor,
        shape=[num_detections]
    discard_image_pixels: If true, discards the image from the result
  Returns:
    The de-serialized TF example augmented with the inferred detections.
  """
  tf_example = tf.train.Example()
  (serialized_example, detected_items) = tf.get_default_session().run([
       serialized_example_tensor, detected_boxes_tensor
   ])
  #pdb.set_trace()
  detected_boxes, detected_scores, detected_classes = yolo2googleout(meta, category_index, detected_items)
  #pdb.set_trace()

  tf_example.ParseFromString(serialized_example)
  feature = tf_example.features.feature
  if(detected_boxes != []):
      try:
          detected_boxes = detected_boxes.T
          feature[standard_fields.TfExampleFields.
                  detection_score].float_list.value[:] = detected_scores
          feature[standard_fields.TfExampleFields.
                  detection_bbox_ymin].float_list.value[:] = detected_boxes[0]
          feature[standard_fields.TfExampleFields.
                  detection_bbox_xmin].float_list.value[:] = detected_boxes[1]
          feature[standard_fields.TfExampleFields.
                  detection_bbox_ymax].float_list.value[:] = detected_boxes[2]
          feature[standard_fields.TfExampleFields.
                  detection_bbox_xmax].float_list.value[:] = detected_boxes[3]
          feature[standard_fields.TfExampleFields.
                  detection_class_label].int64_list.value[:] = detected_classes
      except TypeError:
          tf.logging.info(" a TypeError")
          #pdb.set_trace()

  #print(detected_classes)
  #print(detected_scores)
  #print(detected_boxes.T)

  if discard_image_pixels:
    del feature[standard_fields.TfExampleFields.image_encoded]

  return tf_example

