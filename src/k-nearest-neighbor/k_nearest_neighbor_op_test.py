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

class PointsToVoxelOp(tf.test.TestCase):

    def testBasic(self):
        library_filename = '/lhome/chrrist/workspace/tf-user-ops/build/src/points_to_voxel/libtfops_points-to-voxel.so'
        ptv_module = tf.load_op_library(library_filename)
        points_to_voxel = ptv_module.points_to_voxel

        with self.test_session() as sess:

            # min_z, min_y, min_x, max_z, max_y, max_x
            voxel_extend = tf.constant([0.0, 0.0, 0.0, 1.0, 2.0, 3.0], dtype=tf.float32)
            voxel_number = tf.constant([1, 2, 3], dtype=tf.int32)

            with self.assertRaises(ValueError):
                # wrong point cloud size
                result = points_to_voxel([[[0.1, 0.2, 0.4]]], voxel_extend, voxel_number, 35)
            with self.assertRaises(ValueError):
                # wrong rank (1 instead of 3)
                result = points_to_voxel([0.1, 0.2, 0.4, 1.0], voxel_extend, voxel_number, 35)

            # batch size 1
            # 1st and 2nd point within same voxel (0 0 0 0), third point in (0 0 1 0).
            # fourth outside of grid
            pc = tf.constant([[[0.1, 0.2, 0.4, 1.0], [0.15, 0.22, 0.42, 1.0],
                               [0.5, 1.5, 0.5, 1.0], [-0.5, 0.0, 0.0, 1.0]]], dtype=tf.float32)

            result = sess.run(points_to_voxel(pc, voxel_extend, voxel_number, tf.constant(2, dtype=tf.int32)))

            features = result[0]
            coordinates = result[1]
            counters = result[2]
            point_features = result[3]
            point_to_voxel_mapping = result[4]
            features_to_input_point_mapping = result[5]

            self.assertEqual(features.shape[0], 2)
            self.assertEqual(coordinates.shape[0], 2)
            self.assertEqual(counters.shape[0], 2)

            self.assertEqual(len(point_features.shape), 2)
            self.assertEqual(point_features.shape[0], 3)
            self.assertEqual(point_features.shape[1], 7)

            self.assertEqual(len(point_to_voxel_mapping.shape), 1)
            self.assertEqual(point_to_voxel_mapping.shape[0], 3)

            self.assertEqual(len(features_to_input_point_mapping.shape), 2)
            self.assertEqual(features_to_input_point_mapping.shape[0], 3)
            self.assertEqual(features_to_input_point_mapping.shape[1], 2)

            # -> limit to one point per voxel
            # batch size 1
            # 1st and 2nd point within same voxel (0 0 0 0), third point in (0 0 1 0).
            # fourth outside of grid
            pc = tf.constant([[[0.1, 0.2, 0.4, 1.0], [0.15, 0.22, 0.42, 1.0],
                               [0.5, 1.5, 0.5, 1.0], [-0.5, 0.0, 0.0, 1.0]]], dtype=tf.float32)

            result = sess.run(points_to_voxel(pc, voxel_extend, voxel_number, 1))

            features = result[0]
            coordinates = result[1]
            counters = result[2]
            point_features = result[3]
            point_to_voxel_mapping = result[4]
            features_to_input_point_mapping = result[5]

            self.assertEqual(features.shape[0], 2)
            self.assertEqual(coordinates.shape[0], 2)
            self.assertEqual(counters.shape[0], 2)
            self.assertEqual(counters[0], 1)
            self.assertEqual(counters[1], 1)

            self.assertEqual(len(point_features.shape), 2)
            self.assertEqual(point_features.shape[0], 3)
            self.assertEqual(point_features.shape[1], 7)

            self.assertEqual(len(point_to_voxel_mapping.shape), 1)
            self.assertEqual(point_to_voxel_mapping.shape[0], 3)

            self.assertEqual(len(features_to_input_point_mapping.shape), 2)
            self.assertEqual(features_to_input_point_mapping.shape[0], 3)
            self.assertEqual(features_to_input_point_mapping.shape[1], 2)

            # -> batching
            # batch size 2
            # 5 points, 3 padded with zeros
            # 4 points within voxel extend -> sum of counters
            # 3 different voxels (two in batch 0, one in batch 1)
            # voxels (#batch, #voxel): 0.0, 0.0, 0.1, x, 1.0, -, -, -
            pc = tf.constant([[[0.1, 0.2, 0.4, 1.0], [0.15, 0.22, 0.42, 1.0],
                               [0.5, 1.5, 0.5, 1.0], [-0.5, 0.0, 0.0, 1.0]],
                              [[0.3, 0.3, 0.3, 1.0], [0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
                              ], dtype=tf.float32)

            result = sess.run(points_to_voxel(pc, voxel_extend, voxel_number, 2))

            features = result[0]
            coordinates = result[1]
            counters = result[2]
            point_features = result[3]
            point_to_voxel_mapping = result[4]
            features_to_input_point_mapping = result[5]

            self.assertEqual(features.shape[0], 3)
            self.assertEqual(coordinates.shape[0], 3)
            self.assertEqual(coordinates.shape[1], 4)
            self.assertEqual(counters.ndim, 1)
            self.assertEqual(counters.shape[0], 3)
            self.assertEqual(np.sum(counters), 4)

            # check that counters is a permutation of [1, 1, 2]
            self.assertTrue(np.array_equal(np.asarray([1, 1, 2], dtype=np.int32), np.sort(counters)))

            self.assertEqual(len(point_features.shape), 2)
            self.assertEqual(point_features.shape[0], 4)
            self.assertEqual(point_features.shape[1], 7)

            self.assertEqual(len(point_to_voxel_mapping.shape), 1)
            self.assertEqual(point_to_voxel_mapping.shape[0], 4)

            self.assertEqual(len(features_to_input_point_mapping.shape), 2)
            self.assertEqual(features_to_input_point_mapping.shape[0], 4)
            self.assertEqual(features_to_input_point_mapping.shape[1], 2)

            # test zero value for unused points
            point = [0.1, 0.2, 0.4, 1.0]
            pc = tf.constant([[point]], dtype=tf.float32)

            result = sess.run(points_to_voxel(pc, voxel_extend, voxel_number, 100000000))

            features = result[0]
            first_point = features[0][0]
            unset_points = features[0][1:]

            self.assertAllEqual(unset_points, np.zeros_like(unset_points))
            self.assertAllEqual(first_point, np.array(np.concatenate((point, [0., 0., 0.])), dtype=np.float32))


if __name__ == '__main__':
    tf.test.main()
