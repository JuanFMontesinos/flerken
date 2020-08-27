import cv2
import torch
import numpy as np

from typing import Union, Tuple

from . import _BaseTransformation as BaseTransformation


class Resize(BaseTransformation):
    """Resize the input torch.Tensor to the given size.

        interpolation (str):  INTER_NEAREST - a nearest-neighbor interpolation
        INTER_LINEAR - a bilinear interpolation (used by default)
        INTER_AREA - resampling using pixel area relation. It may be a preferred method for image decimation,
        as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
        INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
        INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood

        https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#resize for detail
    """

    def __init__(self, size: Union[int, Tuple[int, int]],
                 interpolation: [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC,
                                 cv2.INTER_LANCZOS4] = cv2.INTER_LINEAR) -> None:
        self.size: Tuple[int, int] = size
        self.interpolation: str = interpolation

    def __call__(self, input: np.ndarray) -> np.ndarray:  # type: ignore
        return cv2.resize(input, self.size, interpolation=self.interpolation)


class VideoNormalization(BaseTransformation):
    """
    Mean-std Normalization for volumetric data of shape *xC along the dimension c.
    Typical data can be TxHxWxC (volumetric data) or HwWxC
    """

    def __init__(self, epsilon: float = 0.):
        self.epsilon = epsilon

    def __call__(self, array: Union[torch.tensor, np.array]):
        if isinstance(array, torch.Tensor):
            kw = {'dim': 0}
        elif isinstance(array, np.ndarray):
            kw = {'axis': 0}
        else:
            raise TypeError(f'VideoNormalization only accepts torch.Tensor or np.ndarray as input')
        C = array.shape[-1]
        array_ = array.reshape(-1, C)
        mean = array_.mean(**kw)
        std = array_.std(**kw)
        return (array - mean) / (std + self.epsilon)

    def _torch(self):
        tensor = torch.rand(7, 5, 5, 3)
        return tensor, self(tensor)

    def _numpy(self):
        array = np.random.rand(7, 5, 5, 3)
        return array, self(array)

    def _dali(self):
        """
        print(f'Dali shape {dali.shape}')
        print(f' Axis 0 mean:{dali[0,...].mean()}, std {dali[0,...].std()}')
        print(f' Axis 1 mean:{dali[:,0,...].mean()}, std {dali[:,0,...].std()}')
        print(f' Axis 2 mean:{dali[...,0,:].mean()}, std {dali[...,0,:].std()}')
        print(f' Axis 3 mean:{dali[...,0].mean()}, std {dali[...,0].std()}')
        print(abs(dali - n.numpy()).sum())
        """
        from nvidia.dali.pipeline import Pipeline
        import nvidia.dali.fn as fn
        tensor = torch.rand(7, 5, 5, 3)
        pipe = Pipeline(batch_size=1, num_threads=4, device_id=0, exec_pipelined=False, exec_async=False)
        data = fn.external_source(source=[tensor[None, ...]], no_copy=False)
        processed = fn.normalize(data, batch=False, axes=[0, 1, 2])
        fn.vide
        pipe.set_outputs(processed)
        pipe.build()
        out = pipe.run()
        dali = out[0].as_array()[0]
        return dali
