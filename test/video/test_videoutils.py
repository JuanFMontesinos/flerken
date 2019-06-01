import unittest

from flerken.video import utils


class TestFramework(unittest.TestCase):
    def test_get_metadata(self):
        time, fps = utils.get_duration_fps('./test/test.mp4', display='s')
        self.assertEqual(fps, 25.0)
        self.assertEqual(time, 18.65)

        time, fps = utils.get_duration_fps('./test/test.mp4', display='ms')
        self.assertEqual(time, 18650.0)

        time, fps = utils.get_duration_fps('./test/test.mp4', display='min')
        self.assertEqual(time, 18.65 / 60)

        time, fps = utils.get_duration_fps('./test/test.mp4', display='h')
        self.assertEqual(time, 18.65 / 3600)
