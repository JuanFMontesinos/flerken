import re
import imp
import gc
import os
import torch
import numpy as np
import math
import warnings

from torchvision.io import _video_opt

_HAS_VIDEO_OPT = False

try:
    lib_dir = os.path.join(os.path.dirname(__file__), '..')
    _, path, description = imp.find_module("video_reader", [lib_dir])
    torch.ops.load_library(path)
    _HAS_VIDEO_OPT = True
except (ImportError, OSError):
    pass

try:
    import av

    av.logging.set_level(av.logging.ERROR)
    if not hasattr(av.video.frame.VideoFrame, 'pict_type'):
        av = ImportError("""\
Your version of PyAV is too old for the necessary video operations in torchvision.
If you are on Python 3.5, you will have to build from source (the conda-forge
packages are not up-to-date).  See
https://github.com/mikeboers/PyAV#installation for instructions on how to
install PyAV on your system.
""")
except ImportError:
    av = ImportError("""\
PyAV is not installed, and is necessary for the video operations in torchvision.
See https://github.com/mikeboers/PyAV#installation for instructions on how to
install PyAV on your system.
""")


def _check_av_available():
    if isinstance(av, Exception):
        raise av


def _av_available():
    return not isinstance(av, Exception)


# PyAV has some reference cycles
_CALLED_TIMES = 0
_GC_COLLECTION_INTERVAL = 10


def _read_from_stream(container, start_offset, end_offset, pts_unit, stream, stream_name, transforms):
    global _CALLED_TIMES, _GC_COLLECTION_INTERVAL
    _CALLED_TIMES += 1
    if _CALLED_TIMES % _GC_COLLECTION_INTERVAL == _GC_COLLECTION_INTERVAL - 1:
        gc.collect()

    if pts_unit == 'sec':
        start_offset = int(math.floor(start_offset * (1 / stream.time_base)))
        if end_offset != float("inf"):
            end_offset = int(math.ceil(end_offset * (1 / stream.time_base)))
    else:
        warnings.warn("The pts_unit 'pts' gives wrong results and will be removed in a " +
                      "follow-up version. Please use pts_unit 'sec'.")

    frames = {}
    should_buffer = False
    max_buffer_size = 5
    if stream.type == "video":
        # DivX-style packed B-frames can have out-of-order pts (2 frames in a single pkt)
        # so need to buffer some extra frames to sort everything
        # properly
        extradata = stream.codec_context.extradata
        # overly complicated way of finding if `divx_packed` is set, following
        # https://github.com/FFmpeg/FFmpeg/commit/d5a21172283572af587b3d939eba0091484d3263
        if extradata and b"DivX" in extradata:
            # can't use regex directly because of some weird characters sometimes...
            pos = extradata.find(b"DivX")
            d = extradata[pos:]
            o = re.search(br"DivX(\d+)Build(\d+)(\w)", d)
            if o is None:
                o = re.search(br"DivX(\d+)b(\d+)(\w)", d)
            if o is not None:
                should_buffer = o.group(3) == b"p"
    seek_offset = start_offset
    # some files don't seek to the right location, so better be safe here
    seek_offset = max(seek_offset - 1, 0)
    if should_buffer:
        # FIXME this is kind of a hack, but we will jump to the previous keyframe
        # so this will be safe
        seek_offset = max(seek_offset - max_buffer_size, 0)
    try:
        # TODO check if stream needs to always be the video stream here or not
        container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    except av.AVError:
        # TODO add some warnings in this case
        # print("Corrupted file?", container.name)
        return []
    buffer_count = 0
    try:
        for idx, frame in enumerate(container.decode(**stream_name)):
            if stream.type == "video":
                if transforms is None:
                    frames[frame.pts] = frame.to_rgb().to_ndarray()
                else:
                    frames[frame.pts] = transforms(frame.to_rgb().to_ndarray())
            else:
                frames[frame.pts] = frame.to_ndarray()
            if frame.pts >= end_offset:
                if should_buffer and buffer_count < max_buffer_size:
                    buffer_count += 1
                    continue
                break
    except av.AVError:
        # TODO add a warning
        pass
    # ensure that the results are sorted wrt the pts
    pts = sorted(list(frames.keys()))
    if max(pts)<seek_offset:
        raise BufferError('Offset chosen for seeking %s seems to be out of bounds')
    result = [frames[i] for i in sorted(frames) if start_offset <= i <= end_offset]
    if len(frames) > 0 and start_offset > 0 and start_offset not in frames:
        # if there is no frame that exactly matches the pts of start_offset
        # add the last frame smaller than start_offset, to guarantee that
        # we will have all the necessary data. This is most useful for audio
        preceding_frames = [i for i in frames if i < start_offset]
        if len(preceding_frames) > 0:
            first_frame_pts = max(preceding_frames)
            result.insert(0, frames[first_frame_pts])
    return result, pts


def _align_audio_frames(aframes, audio_frames, ref_start, ref_end):
    start, end = audio_frames[0], audio_frames[-1]
    total_aframes = aframes.shape[1]
    step_per_aframe = (end - start + 1) / total_aframes
    s_idx = 0
    e_idx = total_aframes
    if start < ref_start:
        s_idx = int((ref_start - start) / step_per_aframe)
    if end > ref_end:
        e_idx = int((ref_end - end) / step_per_aframe)
    return aframes[:, s_idx:e_idx]


def read_video(filename, start_pts=0, end_pts=None, pts_unit='pts', read_audio=True, transforms=None):
    """
    Reads a video from a file, returning both the video frames as well as
    the audio frames

    Parameters
    ----------
    filename : str
        path to the video file
    start_pts : int if pts_unit = 'pts', optional
        float / Fraction if pts_unit = 'sec', optional
        the start presentation time of the video
    end_pts : int if pts_unit = 'pts', optional
        float / Fraction if pts_unit = 'sec', optional
        the end presentation time
    pts_unit : str, optional
        unit in which start_pts and end_pts values will be interpreted, either 'pts' or 'sec'. Defaults to 'pts'.

    Returns
    -------
    vframes : Tensor[T, H, W, C]
        the `T` video frames
    aframes : Tensor[K, L]
        the audio frames, where `K` is the number of channels and `L` is the
        number of points
    info : Dict
        metadata for the video and audio. Can contain the fields video_fps (float)
        and audio_fps (int)
    """

    from torchvision import get_video_backend
    if get_video_backend() != "pyav":
        return _video_opt._read_video(filename, start_pts, end_pts, pts_unit)

    _check_av_available()

    if end_pts is None:
        end_pts = float("inf")

    if end_pts < start_pts:
        raise ValueError("end_pts should be larger than start_pts, got "
                         "start_pts={} and end_pts={}".format(start_pts, end_pts))

    info = {}
    video_frames = []
    audio_frames = []

    try:
        container = av.open(filename, metadata_errors='ignore')
    except av.AVError:
        # TODO raise a warning?
        pass
    else:
        if container.streams.video:
            vframes, video_frames = _read_from_stream(container, start_pts, end_pts, pts_unit,
                                                      container.streams.video[0], {'video': 0}, transforms)
            video_fps = container.streams.video[0].average_rate
            # guard against potentially corrupted files
            if video_fps is not None:
                info["video_fps"] = float(video_fps)
        else:
            vframes = []
        if container.streams.audio and read_audio:
            aframes, audio_frames = _read_from_stream(container, start_pts, end_pts, pts_unit,
                                                      container.streams.audio[0], {'audio': 0}, None)
            info["audio_fps"] = container.streams.audio[0].rate
        else:
            aframes = []
        container.close()

    if vframes:
        vframes = torch.as_tensor(np.stack(vframes))
    else:
        # vframes = torch.empty((0, 1, 1, 3), dtype=torch.uint8)
        raise BufferError('Video reader for %s is empty' % filename)
    # if vframes.shape!=(100,224,224,3):
    #     raise BufferError('Video reader for %s is empty' % filename)
    if aframes:
        aframes = np.concatenate(aframes, 1)
        aframes = torch.as_tensor(aframes)
        aframes = _align_audio_frames(aframes, audio_frames, start_pts, end_pts)
    else:
        aframes = torch.empty((1, 0), dtype=torch.float32)

    return vframes, aframes, info
