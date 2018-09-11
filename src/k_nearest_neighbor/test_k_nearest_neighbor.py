#!/usr/bin/env python
# 2018, Patrick Wieschollek <mail@patwie.com>

import numpy as np
import tensorflow as tf
from __init__ import knn

np.random.seed(42)
tf.set_random_seed(42)


class KNNTest(tf.test.TestCase):

  def _forward(self, use_gpu=False, dtype=np.float32):
    #
    ref_points = tf.constant([[1.0, 2.0, 3.0], [0.0, 0.1, 0.1], [-1.0, -1.0, -1.0]], dtype=tf.float32)
    query_points = tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32)

    ref_points2 = tf.tile(tf.constant([[1.0, 1.0, 1.0]]), multiples=[30971, 1])
    ref_points2 = tf.concat([ref_points2, tf.constant([[0.0, 0.0, 0.0]])], axis=0)

    with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu) as sess:
      op = knn(tf.transpose(ref_points), tf.transpose(query_points), 2)
      indices, _ = sess.run(op)

      np.testing.assert_array_equal(np.asarray([[1, 2]], dtype=np.int32), indices)

  def test_forward_float32(self):
    self._forward(use_gpu=True, dtype=np.float32)


if __name__ == '__main__':
  tf.test.main()
