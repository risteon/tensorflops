#!/usr/bin/env python

import os
import csv
import numpy as np
import tensorflow as tf
from __init__ import knn

np.random.seed(42)
tf.set_random_seed(42)


class KNNTest(tf.test.TestCase):

    def _data_input(self, directory):
        def get_array(filename, dtype):
            with open(filename, newline='') as csvfile:
                csvreader = csv.reader(csvfile, delimiter='\n', quotechar='|')
                return np.asarray([r for r in csvreader]).astype(dtype)

        with open(os.path.join(directory, 'meta.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            meta = dict(reader)
            meta = {k: int(v) for k, v in meta.items()}

        np_dist = np.reshape(get_array(os.path.join(directory, 'knn_dist.csv'), np.float32),
                             newshape=[-1, meta['k']])
        np_index = np.reshape(get_array(os.path.join(directory, 'knn_index.csv'), np.int32),
                              newshape=[-1, meta['k']])
        np_points = np.reshape(get_array(os.path.join(directory, 'points.csv'), np.float32),
                               newshape=[meta['dim'], -1])
        np_query = np.reshape(get_array(os.path.join(directory, 'query_points.csv'), np.float32),
                              newshape=[meta['dim'], -1])

        tf_points = tf.constant(np_points, dtype=tf.float32)
        tf_query = tf.constant(np_query, dtype=tf.float32)

        with self.test_session(use_gpu=True, force_gpu=True) as sess:
            op = knn(tf_points, tf_query, meta['k'])
            indices, distances = sess.run(op)

            self.assertLess(np.sum(indices != np_index) / indices.size, 4e-05)
            self.assertEqual(np.sum(np.isclose(distances, np_dist, atol=0.00001)), distances.size)

    def test_small(self):
        ref_points = tf.constant([[1.0, 2.0, 3.0], [0.0, 0.1, 0.1], [-1.0, -1.0, -1.0]],
                                 dtype=tf.float32)
        query_points = tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32)

        with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu) as sess:
            op = knn(tf.transpose(ref_points), tf.transpose(query_points), 2)
            indices, _ = sess.run(op)
            np.testing.assert_array_equal(np.asarray([[1, 2]], dtype=np.int32), indices)

    def test_with_data(self):
        self._data_input(os.path.join('test', 'test1'))
        self._data_input(os.path.join('test', 'test2'))
        self._data_input(os.path.join('test', 'test3'))


if __name__ == '__main__':
    tf.test.main()
