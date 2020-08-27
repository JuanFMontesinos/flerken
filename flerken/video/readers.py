import os
from typing import Union

import cv2

try:
    import imageio
except ImportError:
    pass
try:
    import skvideo
except ImportError:
    pass

import numpy as np

from .io import read_video

__all__ = ['ImageIOSampler', 'CV2VideoSampler', 'TorchVideoReader']


class CV2VideoSampler:
    def __init__(self, processor=None):
        self.processor = processor

    def __call__(self, path, offset, length):
        if not os.path.isfile(path):
            raise FileNotFoundError('Video clip at %s were not found' % path)
        frames = []
        flags = []
        cap = cv2.VideoCapture(path)
        if offset is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, offset)
        for i in range(length):
            if not cap.isOpened():
                raise ConnectionAbortedError('OpenCV failed to open %s' % path)
            flag, frame = cap.read()
            if self.processor is not None:
                frame = self.processor(frame)
            frames.append(frame)
            flags.append(flag)
        if any(flags) == False:
            raise ValueError('Frames captured at %s are wrong' % path)
        cap.release()
        return np.stack(frames)


class ImageIOSampler:
    def __init__(self, processor=None):
        self.processor = processor

    def __call__(self, path: str, offset: Union[None, int], length: int):
        if not os.path.isfile(path):
            raise FileNotFoundError('Video clip at %s were not found' % path)
        frames = []

        reader = imageio.get_reader(path)
        if offset is not None:
            reader.set_image_index(offset)
        for _ in range(length):
            frame = reader.get_next_data()
            if self.processor is not None:
                frame = self.processor(frame)
            frames.append(frame)
        return np.stack(frames)


class TorchVideoReader:
    def __init__(self, framerate, processor, read_audio):
        self.pts_unit = 'sec'
        self.fps = framerate
        self.processor = processor
        self.read_audio = read_audio

    def __call__(self, path, offset, length):
        return read_video(path, start_pts=(offset + 1) / self.fps,
                          end_pts=(offset + length) / self.fps,
                          pts_unit=self.pts_unit,
                          transforms=self.processor,
                          read_audio=self.read_audio)
