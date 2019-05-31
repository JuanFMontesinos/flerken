import os
import re
import subprocess

__all__ = ['get_video_metadata']


def get_video_metadata(filename, display=None):
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
    info_lines = [x for x in output if "Duration" in x or "Stream" in x]
    duration_line = [x for x in info_lines if "Duration" in x]
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
    return time, fps
