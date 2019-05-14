import types
from functools import partial
import sys


class Celsius:
    def __init__(self):
        self._caca = 17

    @staticmethod
    def coger(self, name):
        print('get')
        return getattr(self, name)

    @staticmethod
    def poner(self, item, name):
        print('set')
        return setattr(self, name, item)

    def hola(self):
        print('hola')

    def adios(self):
        print('adios')

    def __set_property_attr__(self, name, value):
        setattr(self, '_' + name, value)

    def __enter__(self):
        print('enters')
        self.pedro = self.adios

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pedro = self.hola
        self.mundial = 21

    def __set_property__(self, name, set_function_name, get_function_name):
        set_function_name_base = set_function_name
        get_function_name_base = get_function_name
        set_function_name += '_' + name
        get_function_name += '_' + name
        setattr(self, set_function_name, partial(getattr(self, set_function_name_base), name=name))
        setattr(self, get_function_name, partial(getattr(self, get_function_name_base), name=name))
        setattr(self.__class__, name, property(getattr(self, get_function_name), getattr(self, set_function_name)))

    def test(self):
        with inference(self):
            print('with')
            self.pedro()
        print('without')
        self.pedro()

    pedro = hola


class inference(object):
    def __new__(cls, obj):
        print('obj')
        return obj


a = Celsius()

a.__set_property__('caca', 'poner', 'coger')

a.test()


class caca:
    def __init__(self, temperature=0):
        self._temperature = temperature

    def to_fahrenheit(self):
        return (self.temperature * 1.8) + 32

    @property
    def temperature(self):
        print("Getting value")
        print(sys._getframe(1).f_code.co_name)
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        if value < -273:
            raise ValueError("Temperature below -273 is not possible")
        print("Setting value")
        print(sys._getframe(1))
        self._temperature = value


w = caca()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, hist=False):
        self.track_hist = hist
        self.reset()
        self.called = False

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        if self.track_hist:
            self.hist = []

    def update(self, val, n=1):
        self.called = True
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.track_hist:
            self.hist.append(val)


class Timers(object):
    def __init__(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_loss = AverageMeter(hist=True)
        self.train_epoch_loss = AverageMeter(hist=True)
        self.val_loss = AverageMeter(hist=True)
        self.val_epoch_loss = AverageMeter(hist=True)
        self.test_loss = AverageMeter(hist=True)
        self.test_epoch_loss = AverageMeter(hist=True)
        self.end = time.time()
        self.enabled = 0x111111

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, mask):
        mask = '0x{:06x}'.format(mask)[2:]  # Converts mask to string and removes 0x
        self.train_loss.enabled = bool(mask[0])
        self.train_epoch_loss.enabled = bool(mask[1])
        self.val_loss.enabled = bool(mask[2])
        self.val_epoch_loss.enabled = bool(mask[3])
        self.test_loss.enabled = bool(mask[4])
        self.test_epoch_loss.enabled = bool(mask[5])

    def update_loss(self):
        self.train_loss.update(self.loss)

    def update_timed(self):
        self.data_time.update(time.time() - self.end)

    def update_timeb(self):
        self.batch_time.update(time.time() - self.end)

    def update_epoch(self):
        value = np.mean(self.train_loss.hist)
        self.train_epoch_loss.update(value)
        return value

    def update_val_epoch(self):
        value = np.mean(self.val_loss.hist)
        self.val_epoch_loss.update(value)
        return value

    def update_test_epoch(self):
        value = np.mean(self.test_loss.hist)
        self.test_epoch_loss.update(value)
        return value

    def print_logger(self, t, j, iterations, logger):
        logger.info('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
            t, j, iterations, batch_time=self.batch_time,
            data_time=self.data_time, loss=self.train_loss))

    def epoch_logger(self, current_epoch, total_epochs, epoch_loss, logger):
        logger.info('Epoch: [{0}/{1}]\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss ({loss:.4f})'.format(
            current_epoch, total_epochs, data_time=self.data_time, loss=epoch_loss))

    def is_best(self):
        if self.val_epoch_loss.called:
            return (np.mean(self.val_loss.hist) <= np.min(self.val_epoch_loss.hist))
        else:
            return (np.mean(self.train_loss.hist) <= np.min(self.train_epoch_loss.hist))
