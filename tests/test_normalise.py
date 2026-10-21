"""
test_normalise.py - 12 Unit tests for pose landmark normalization (pose_normalise.py).
Executable via both `pytest -q test_normalise.py` and `python test_normalise.py`.
"""

import unittest
import numpy as np
from pose_normalise import (
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    normalise,
)


def get_sample_landmarks():
  """Generates a realistic synthetic 33-landmark pose array (33, 4)."""
  np.random.seed(42)
  lm = np.zeros((33, 4), dtype=float)
  lm[:, :3] = np.random.uniform(-1.0, 1.0, size=(33, 3))
  lm[:, 3] = 0.9  # visibility score

  # Define realistic relative positions for key joints
  lm[LEFT_SHOULDER, :3] = np.array([-0.3, 0.5, 0.0])
  lm[RIGHT_SHOULDER, :3] = np.array([0.3, 0.5, 0.0])
  lm[LEFT_HIP, :3] = np.array([-0.2, -0.2, 0.0])
  lm[RIGHT_HIP, :3] = np.array([0.2, -0.2, 0.0])
  return lm


class TestPoseNormalise(unittest.TestCase):

  def setUp(self):
    self.sample = get_sample_landmarks()

  # 1. Test shape preservation for single frame (33, 4)
  def test_01_shape_preservation_single_frame(self):
    norm = normalise(self.sample)
    self.assertEqual(norm.shape, (33, 4))

  # 2. Test shape preservation for batch input (N, 33, 4)
  def test_02_shape_preservation_batch(self):
    batch = np.stack([self.sample, self.sample * 1.5])
    norm = normalise(batch)
    self.assertEqual(norm.shape, (2, 33, 4))

  # 3. Test 3D input format (33, 3)
  def test_03_support_3d_coordinates_only(self):
    lm_3d = self.sample[:, :3]
    norm = normalise(lm_3d)
    self.assertEqual(norm.shape, (33, 3))

  # 4. Test translation: Hip center is mapped to origin (0, 0, 0)
  def test_04_translation_centering_at_origin(self):
    norm = normalise(self.sample)
    hip_mid = (norm[LEFT_HIP, :3] + norm[RIGHT_HIP, :3]) / 2.0
    np.testing.assert_allclose(hip_mid, [0.0, 0.0, 0.0], atol=1e-6)

  # 5. Test scale invariance: Scaling raw coordinates yields identical normalized coordinates
  def test_05_scale_invariance(self):
    norm_orig = normalise(self.sample)
    shifted_scaled = np.copy(self.sample)
    shifted_scaled[:, :3] = (self.sample[:, :3] * 3.5) + np.array(
        [10.0, -5.0, 2.0]
    )
    norm_shifted = normalise(shifted_scaled)
    np.testing.assert_allclose(norm_orig[:, :3], norm_shifted[:, :3], atol=1e-5)

  # 6. Test normalized torso length equals reference length (1.0)
  def test_06_unit_torso_length(self):
    norm = normalise(self.sample, reference_torso_length=1.0)
    sh_center = (norm[LEFT_SHOULDER, :3] + norm[RIGHT_SHOULDER, :3]) / 2.0
    hip_center = (norm[LEFT_HIP, :3] + norm[RIGHT_HIP, :3]) / 2.0
    torso_length = np.linalg.norm(sh_center - hip_center)
    self.assertAlmostEqual(torso_length, 1.0, places=5)

  # 7. Test visibility column (4th column) is preserved without alteration
  def test_07_visibility_preserved(self):
    vis_before = self.sample[:, 3].copy()
    norm = normalise(self.sample)
    np.testing.assert_array_equal(norm[:, 3], vis_before)

  # 8. Test idempotency: Normalizing an already normalized pose produces identical results
  def test_08_idempotency(self):
    norm_once = normalise(self.sample)
    norm_twice = normalise(norm_once)
    np.testing.assert_allclose(norm_once, norm_twice, atol=1e-6)

  # 9. Test NaN handling: Data with NaNs does not throw an unexpected exception
  def test_09_nan_handling(self):
    lm_nan = np.copy(self.sample)
    lm_nan[0, :3] = np.nan  # Set nose to NaN
    norm = normalise(lm_nan)
    self.assertTrue(np.isnan(norm[0, :3]).all())
    self.assertEqual(norm.shape, (33, 4))

  # 10. Test all-zero input handling
  def test_10_zero_landmarks_handling(self):
    zero_lm = np.zeros((33, 4), dtype=float)
    norm = normalise(zero_lm)
    np.testing.assert_array_equal(norm, zero_lm)

  # 11. Test custom reference torso length scaling
  def test_11_custom_reference_torso_length(self):
    norm_2 = normalise(self.sample, reference_torso_length=2.0)
    sh_center = (norm_2[LEFT_SHOULDER, :3] + norm_2[RIGHT_SHOULDER, :3]) / 2.0
    hip_center = (norm_2[LEFT_HIP, :3] + norm_2[RIGHT_HIP, :3]) / 2.0
    torso_length = np.linalg.norm(sh_center - hip_center)
    self.assertAlmostEqual(torso_length, 2.0, places=5)

  # 12. Test invalid shape raises ValueError
  def test_12_invalid_shape_raises_value_error(self):
    invalid_lm = np.zeros((20, 4))
    with self.assertRaises(ValueError):
      normalise(invalid_lm)


if __name__ == "__main__":
  unittest.main()