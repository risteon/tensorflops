# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for custom user ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf
import numpy as np

class KNearestNeighborOp(tf.test.TestCase):

    def testBasic(self):
        library_filename = '/lhome/chrrist/workspace/tensorflops/debug/src/k-nearest-neighbor/tfops_k-nearest-neighbor.so.0.1'

        knn_module = tf.load_op_library(library_filename)
        knn = knn_module.k_nearest_neighbor

        with self.test_session() as sess:

            # min_z, min_y, min_x, max_z, max_y, max_x
            ref_points = tf.constant([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=tf.float32)
            query_points = tf.constant([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=tf.float32)

            indices, distances = knn(ref_points, query_points, tf.constant(2, dtype=tf.int32))

            np_indices, np_distances = sess.run([indices, distances])


if __name__ == '__main__':
    tf.test.main()
