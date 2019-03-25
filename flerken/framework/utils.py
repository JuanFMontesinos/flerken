__author__ = "Juan Montesinos"
__version__ = "0.1"
__maintainer__ = "Juan Montesinos"
__email__ = "juanfelipe.montesinos@upf.edu"
__status__ = "Prototype"
import logging
import time
import numpy as np

def setup_logger(logger_name, log_file, level=logging.INFO,FORMAT='%(message)s',writemode='w',to_console=True):
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
    def __init__(self,hist = False):
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
class timers(object):
    def __init__(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.batch_loss = AverageMeter(hist =True)
        self.epoch_loss = AverageMeter(hist = True)
        self.end = time.time()
    def update_loss(self):
        self.batch_loss.update(self.loss)
    def update_timed(self):
        self.data_time.update(time.time() - self.end)
    def update_timeb(self):
        self.batch_time.update(time.time() - self.end)
    def update_epoch(self):
        self.epoch_loss.update(np.mean(self.batch_loss.hist))
    def print_logger(self,t,j,iterations,logger):
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
            t, j, iterations, batch_time=self.batch_time,
            data_time=self.data_time, loss=self.batch_loss))
    def epoch_logger(self, current_epoch,total_epochs,epoch_loss, logger):
        logger.info('Epoch: [{0}/{1}]\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss ({loss:.4f})'.format(
            current_epoch,total_epochs, data_time=self.data_time, loss=epoch_loss))

    def test_logger(self, batch_loss, logger):
        logger.info('Loss ({loss:.4f})'.format(loss=batch_loss))

    def is_best(self):
        return (np.mean(self.batch_loss.hist) <= np.min(self.epoch_loss.hist))
