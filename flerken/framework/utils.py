__author__ = "Juan Montesinos"
__version__ = "0.1"
__maintainer__ = "Juan Montesinos"
__email__ = "juanfelipe.montesinos@upf.edu"
__status__ = "Prototype"
import logging
import time
import numpy as np



def setup_logger(logger_name, log_file, level=logging.WARNING, FORMAT='%(message)s', writemode='w', to_console=False):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(FORMAT)
    fileHandler = logging.FileHandler(log_file, mode=writemode)
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    if to_console:
        l.addHandler(streamHandler)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, hist=False):
        self.track_hist = hist
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        if self.track_hist:
            self.hist = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.track_hist:
            self.hist.append(val)



