import math
import unittest

from cpu_reference import (
    block_scale_ue4m3,
    decode_candidate_e0m3,
    decode_e2m1,
    dot_reference,
    matmul_reference,
)


class DecodeTest(unittest.TestCase):
    def test_e2m1_all_nibbles(self):
        expected = (
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        )
        self.assertEqual(tuple(decode_e2m1(i) for i in range(16)), expected)
        self.assertEqual(math.copysign(1.0, decode_e2m1(0x8)), -1.0)

    def test_candidate_e0m3_all_nibbles(self):
        expected = tuple(float(i) for i in range(8)) + tuple(-float(i) for i in range(8))
        self.assertEqual(tuple(decode_candidate_e0m3(i) for i in range(16)), expected)
        self.assertEqual(math.copysign(1.0, decode_candidate_e0m3(0x8)), -1.0)

    def test_rejects_invalid_nibbles(self):
        for invalid in (-1, 16):
            with self.assertRaises(ValueError):
                decode_e2m1(invalid)
        with self.assertRaises(TypeError):
            decode_e2m1(True)


class ScaleTest(unittest.TestCase):
    def test_ue4m3_boundaries(self):
        self.assertEqual(block_scale_ue4m3(0x00), 0.0)
        self.assertEqual(block_scale_ue4m3(0x01), 2.0**-9)
        self.assertEqual(block_scale_ue4m3(0x38), 1.0)
        self.assertEqual(block_scale_ue4m3(0x78), 256.0)
        self.assertEqual(block_scale_ue4m3(0x7E), 448.0)
        self.assertTrue(math.isnan(block_scale_ue4m3(0x7F)))

    def test_ue4m3_does_not_silently_mask_tag(self):
        with self.assertRaises(ValueError):
            block_scale_ue4m3(0xB8)


class ArithmeticTest(unittest.TestCase):
    def test_dot_with_scale_and_accumulator(self):
        got = dot_reference(
            [0x1] * 64,
            [0x7] * 64,
            a_format="e2m1",
            b_format="e0m3_candidate",
            a_scale=0.5,
            b_scale=2.0,
            accumulator=3.0,
        )
        self.assertEqual(got, 227.0)

    def test_matmul(self):
        got = matmul_reference(
            [[0x2, 0x4], [0xA, 0x1]],
            [[0x1, 0x9], [0x2, 0x3]],
            a_format="e2m1",
            b_format="e0m3_candidate",
            accumulator=[[1.0, 1.0], [2.0, 2.0]],
        )
        self.assertEqual(got, [[6.0, 6.0], [2.0, 4.5]])

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            matmul_reference(
                [[0x1, 0x2]],
                [[0x1]],
                a_format="e2m1",
                b_format="e2m1",
            )


if __name__ == "__main__":
    unittest.main()
