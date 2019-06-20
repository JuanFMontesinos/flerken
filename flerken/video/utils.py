import os
import re
import subprocess
import multiprocessing as mp
import warnings

from torchtree import Directory_Tree

from . import allowed_formats

__all__ = ['get_duration_fps']


def get_duration_fps(filename, display):
    """
    Wraps ffprobe to get file duration and fps

    :param filename: str, Path to file to be evaluate
    :param display: ['ms','s','min','h'] Time format miliseconds, sec, minutes, hours.
    :return: tuple(time, fps) in the mentioned format
    """

    def ffprobe2ms(time):
        cs = int(time[-2::])
        s = int(os.path.splitext(time[-5::])[0])
        idx = time.find(':')
        h = int(time[0:idx - 1])
        m = int(time[idx + 1:idx + 3])
        return [h, m, s, cs]

    # Get length of video with filename
    time = None
    fps = None
    result = subprocess.Popen(["ffprobe", str(filename)],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = [str(x) for x in result.stdout.readlines()]
    info_lines = [x for x in output if "Duration:" in x or "Stream" in x]
    duration_line = [x for x in info_lines if "Duration:" in x]
    fps_line = [x for x in info_lines if "Stream" in x]
    if duration_line:
        duration_str = duration_line[0].split(",")[0]
        pattern = '\d{2}:\d{2}:\d{2}.\d{2}'
        dt = re.findall(pattern, duration_str)[0]
        time = ffprobe2ms(dt)
    if fps_line:
        pattern = '(\d{2})(.\d{2})* fps'
        fps_elem = re.findall(pattern, fps_line[0])[0]
        fps = float(fps_elem[0] + fps_elem[1])
    if display == 's':
        time = time[0] * 3600 + time[1] * 60 + time[2] + time[3] / 100.0
    elif display == 'ms':
        time = (time[0] * 3600 + time[1] * 60 + time[2] + time[3] / 100.0) * 1000
    elif display == 'min':
        time = (time[0] * 3600 + time[1] * 60 + time[2] + time[3] / 100.0) / 60
    elif display == 'h':
        time = (time[0] * 3600 + time[1] * 60 + time[2] + time[3] / 100.0) / 3600
    return (time, fps)


def reencode_25_interpolate(video_path: str, dst_path: str, *args, **kwargs):
    input_options = ['-y']
    output_options = ['-r', '25']
    return apply_single(video_path, dst_path, input_options, output_options)


def reencode_30_interpolate(video_path: str, dst_path: str, *args, **kwargs):
    input_options = ['-y']
    output_options = ['-r', '30']
    return apply_single(video_path, dst_path, input_options, output_options)


def apply_single(video_path: str, dst_path: str, input_options: list, output_options: list):
    """
    Runs ffmpeg for the following format for a single input/output:
        ffmpeg [input options] -i input [output options] output


    :param video_path: str Path to input video
    :param dst_path: str Path to output video
    :param input_options: List[str] list of ffmpeg options ready for a Popen format
    :param output_options: List[str] list of ffmpeg options ready for a Popen format
    :return: None
    """
    assert os.path.isfile(video_path)
    assert os.path.isdir(os.path.dirname(dst_path))
    result = subprocess.Popen(["ffmpeg", *input_options, '-i', video_path, *output_options, dst_path],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = result.stdout.read()
    stderr = result.stdout.read()
    if stdout is not None:
        print(stdout.decode("utf-8"))
    if stderr is not None:
        print(stderr.decode("utf-8"))


def apply_tree(root, dst, input_options=list(), output_options=list(), multiprocessing=0, fn=apply_single):
    """
    Applies ffmpeg processing for a given directory tree for whatever which fits this format:
        ffmpeg [input options] -i input [output options] output
    System automatically checks if files in directory are ffmpeg compatible
    Results will be stored in a replicated tree with same structure and filenames


    :param root: Root directory in which files are stored
    :param dst: Destiny directory in which to store results
    :param input_options: list[str] ffmpeg input options in a subprocess format
    :param output_options: list[str] ffmpeg output options in a subprocess format
    :param multiprocessing: int if 0 disables multiprocessin, else enables multiprocessing with that amount of cores.
    :param fn: funcion to be used. By default requires I/O options.
    :return: None
    """
    formats = allowed_formats()  # List of ffmpeg compatible formats
    tree = Directory_Tree(root)  # Directory tree
    if not os.path.exists(dst):
        os.mkdir(dst)
    tree.clone_tree(dst)  # Generates new directory structure (folders)

    # Python Multiproceesing mode
    if multiprocessing > 0:
        pool = mp.Pool(multiprocessing)
        results = [pool.apply(fn,
                              args=(i_path, o_path, input_options, output_options))
                   for i_path, o_path in zip(tree.paths(root), tree.paths(dst)) if
                   os.path.splitext(i_path)[1][1:] in formats]
        pool.close()
    else:
        for i_path, o_path in zip(tree.paths(root), tree.paths(dst)):
            if os.path.splitext(i_path)[1][1:] in formats:
                fn(i_path, o_path, input_options, output_options)


def quirurgical_extractor(video_path, dst_frames, dst_audio, t, n_frames, T, size=None, sample_rate=None):
    """
    FFMPEG wraper which extracts, from an instant t onwards, n_frames (jpg) and T seconds of audio (wav).
    Optionally performs frame resizing and audio resampling.


    :param video_path: str path to input video
    :param dst_frames: str path to folder in which to store outgoing frames (If doesn't exist will be created)
    :param dst_audio: str path to folder in which to store outgoing audio (If doesn't exist will be created)
    :param t: Initial time from which start the extraction
    :param n_frames: Amount of frames to be extracted from t
    :param T: amount of seconds of audio to be extracted from t
    :param size: str (optional) frame resizing (ej: '256x256')
    :param sample_rate: (optional) audio resampling
    :return: List[str] line-wise console output
    """

    #    dst_frames = os.path.join(dst_frames, '{0:05d}'.format(t))
    if not os.path.exists(dst_audio):
        os.makedirs(dst_audio)
    dst_audio = os.path.join(dst_audio, '{0:05d}.wav'.format(t))
    if not os.path.exists(dst_frames):
        os.makedirs(dst_frames)
    stream = ['ffmpeg', '-y', '-ss', str(t), '-i', video_path]
    if size is not None:
        stream.extend(['-s', size])
    stream.extend(['-frames:v', str(n_frames), dst_frames + '/%02d.jpg', '-vn', '-ac', '1'])
    if sample_rate is not None:
        stream.extend(['-ar', str(sample_rate)])
    stream.extend(['-t', str(T), dst_audio])

    result = subprocess.Popen(stream,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = result.stdout.read()
    stderr = result.stdout.read()
    if stdout is not None:
        print(stdout.decode("utf-8"))
    if stderr is not None:
        print(stderr.decode("utf-8"))


def quirurgical_extractor_tree(root, dst, n_frames, T, size=None, sample_rate=None, multiprocessing=0,
                               stamp_generator_fn=None, formats=None):
    """
    Quirurgical extractor over a directory tree. Clones directory tree in dst folder twice, once for audio and another
    time for frames. For each recording there will be a folder with the same name in dst directory containing audio segm
    ments in case of audio or folders with frames in case of video.


    :param root: Original path of recordings
    :param dst: Destiny path
    :param n_frames: Amount of frames to get for each instant
    :param T: Amount of time (seconds) of audio to get for each instant
    :param size: Optinal image resizing for all frames
    :param sample_rate: Optinal audio resampling for all tracks
    :param multiprocessing: Whether to use or not multiprocessing, By default, 0 , disables MP.
     Else selects amount of cores
    :param stamp_generator_fn: By default function generate time stamps in wich to cut. It is possible to pass
    custom function that takes as input recording path and returns time in seconds (integer). As suggestion you may
    use a dictionary by passing dict.get method as generator function.
    :param formats: List of strings indicating allowed formats. Only files with chosen formats will be processed. By
    default this list is taken from ffmpeg -formats.
    """

    def stamp_generator(video_path):
        time, fps = get_duration_fps(video_path, 's')
        segments = range(0, int(time) - T, T)
        if T * fps != n_frames:
            warnings.warn('Warning: n_frames != fps * T')
        return segments

    def path_stamp_generator(path, gen, T):
        for x in gen:
            yield (os.path.join(path, '{0:05d}to{1:05d}'.format(x, x + T)), x)

    if stamp_generator_fn is None:
        stamp_generator_fn = stamp_generator
    if formats is None:
        formats = allowed_formats()  # List of ffmpeg compatible formats
    tree = Directory_Tree(root)  # Directory tree
    """
    Tree generates an object-based tree of the directory treem, where folders are nodes and parameters are files.
    The idea is to replicate this directory tree for audio files and video files in audio_root and video_root 
    respectively.
    """
    audio_root = os.path.join(dst, 'audio')
    video_root = os.path.join(dst, 'frames')
    if not os.path.exists(audio_root):
        os.makedirs(audio_root)  # Create directory
    if not os.path.exists(video_root):
        os.makedirs(video_root)
    tree.clone_tree(audio_root)  # Clone directory tree structure in for each one
    tree.clone_tree(video_root)

    """
    There is an important concept to get:
    For each video, we will obtain S segments, nevertheless, each segment contains different amount of files depending
    on if it's audio or frames.

    For frames we get S*n_frames frames, which are distributed in S folders. This means that for each video there will 
    be S folders containing n_frames frames.

    On contrary, for audio we will obtain S files, therefore there will be  a single folder per video containing S files

    Be aware video folders have to be created dynamically depending on S, meanwhile audio directory tree can be
    precomputed based on amount of recording.

    """

    """
    These generator generate relative paths to all files existing in the directory. Therefore we obtain a paths to all
    recordings, paths to audio folders (1 folder per recording) and path to video folders. We still have to generate S
    video subfolders for all the segments
    """
    i_path_generator = tree.named_parameters(root)  # Generator of input (path,format)
    a_path_generator = tree.named_parameters(audio_root)  # Generator of audio-wise output paths
    v_path_generator = tree.named_parameters(video_root)  # Generator of video-wise output paths

    failure_list = []
    for i_path, a_path, v_path in zip(i_path_generator, a_path_generator, v_path_generator):
        # This for loop provides needed paths recording-wise
        try:
            if i_path[1][1:] in formats:  # Check file has a valid ffmpeg format.
                stamp_gen = stamp_generator_fn(i_path[0] + i_path[1])
                path_stamp_gen = path_stamp_generator(v_path[0], stamp_gen, T)
                """
                Generator that calculate amount of segments,t,  and dynamically generates S video subfolder paths.
                Obviously, this generator provides t segments and t subfolder names.
                """
                if multiprocessing == 0:
                    for path, t in path_stamp_gen:
                        quirurgical_extractor(i_path[0] + i_path[1], path, a_path[0], t, n_frames, T, size, sample_rate)
                else:
                    pool = mp.Pool(multiprocessing)
                    results = [pool.apply(quirurgical_extractor,
                                          args=(i_path[0] + i_path[1], path, a_path[0], t, n_frames, T, size, sample_rate))
                               for path, t in path_stamp_gen]
                    pool.close()
        except:
            failure_list.append(i_path)
    print('Following videos have failed (from python side):\n')
    for v in failure_list:
        print('{0} \n'.format(v))
    return failure_list
