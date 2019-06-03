import subprocess as _subprocess


def allowed_formats():
    result = _subprocess.Popen(["ffmpeg", '-formats'],
                               stdout=_subprocess.PIPE, stderr=_subprocess.STDOUT)
    output = [str(x).split()[1:3] for x in result.stdout.readlines()]
    remake = {}
    for i, x in enumerate(output):
        if x[0] == 'E' or x[0] == 'D' or x[0] == 'DE' or x[0] == 'ED':
            for element in x[1].split(','):
                remake.update({element: x[0]})
    return remake


from . import utils
