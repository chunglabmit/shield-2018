import contextlib
import json
import numpy as np
import os
import pickle
import shield_2018.segmentation as seg
import tempfile
import tifffile
import unittest


class TestSegmentationParams(unittest.TestCase):
    def test_can_pickle(self):
        # Make sure pickling for multiprocessing yields a reasonably-sized
        # object
        params = seg.SegmentationParameters()
        params.stackmem = seg.SharedMemory((1000, 1000, 100), np.uint16)
        params.dogmem = seg.SharedMemory((1000, 1000, 100), np.float32)
        params.curvmem = seg.SharedMemory((1000, 1000, 100), np.float32)
        sp_pickle = pickle.dumps(params)
        self.assertLess(len(sp_pickle), 10000)


@contextlib.contextmanager
def make_stack(stack):
    directory = tempfile.mkdtemp()
    for i, plane in enumerate(stack):
        tifffile.imsave(os.path.join(directory, "img_%04d.tiff" % i), plane,
                        compress=3)
    yield os.path.join(directory, "img_*.tiff")
    for i in range(len(stack)):
        os.remove(os.path.join(directory, "img_%04d.tiff" % i))
    os.rmdir(directory)


class TestDoSegmentation(unittest.TestCase):

    def test_nothing(self):
        with make_stack(np.zeros((100, 100, 100), np.uint16)) as glob_expr:
            with tempfile.NamedTemporaryFile(suffix=".json") as output_tf:
                seg.do_segmentation(glob_expr, output_tf.name)
                coords = json.load(output_tf)
                self.assertEqual(len(coords), 0)

    def test_one(self):
        z, y, x = np.mgrid[-20:80, -30:70, -40:60]
        stack = np.zeros((100, 100, 100), np.uint16)
        stack[x*x + y*y + z*z < 25] = 1000
        with make_stack(stack) as glob_expr:
            with tempfile.NamedTemporaryFile(suffix=".json") as output_tf:
                seg.do_segmentation(glob_expr, output_tf.name)
                coords = json.load(output_tf)
                self.assertEqual(len(coords), 1)
                self.assertAlmostEqual(coords[0][0], 40, delta=2)
                self.assertAlmostEqual(coords[0][1], 30, delta=2)
                self.assertAlmostEqual(coords[0][2], 20, delta=2)

    def test_adaptive(self):
        z, y, x = np.mgrid[-20:80, -30:70, -40:60]
        stack = np.zeros((100, 100, 100), np.uint16)
        stack[x * x + y * y + z * z < 25] = 1000
        with make_stack(stack) as glob_expr:
            with tempfile.NamedTemporaryFile(suffix=".json") as output_tf:
                seg.do_segmentation(glob_expr, output_tf.name,
                                    use_adaptive_threshold=True)
                coords = json.load(output_tf)
                self.assertEqual(len(coords), 1)
                self.assertAlmostEqual(coords[0][0], 40, delta=2)
                self.assertAlmostEqual(coords[0][1], 30, delta=2)
                self.assertAlmostEqual(coords[0][2], 20, delta=2)

    def test_seeds(self):
        z, y, x = np.mgrid[-20:80, -30:70, -40:60]
        stack = np.zeros((100, 100, 100), np.uint16)
        stack[x*x + y*y + z*z < 25] = 1000
        with make_stack(stack) as glob_expr:
            with tempfile.NamedTemporaryFile(suffix=".json") as output_tf:
                seg.do_segmentation(glob_expr, output_tf.name,
                                    use_seed_centers=True)
                coords = json.load(output_tf)
                self.assertEqual(len(coords), 1)
                self.assertAlmostEqual(coords[0][0], 40, delta=2)
                self.assertAlmostEqual(coords[0][1], 30, delta=2)
                self.assertAlmostEqual(coords[0][2], 20, delta=2)



if __name__ == '__main__':
    unittest.main()
