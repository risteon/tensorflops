# 2018, Patrick Wieschollek <mail@patwie.com>

# manually generated file
import tensorflow as tf
import os
from tensorflow.python.framework import ops

__all__ = ['knn']

#path = os.path.join(os.path.dirname(__file__), 'matrix_add_op.so')
path = '/lhome/chrrist/workspace/tensorflops/release/src/k_nearest_neighbor/tfops_k_nearest_neighbor.so.0.1'
_knn_module = tf.load_op_library(path)

knn = _knn_module.k_nearest_neighbor
