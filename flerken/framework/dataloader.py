from threading import Thread
from time import sleep
import queue
import time
import torch


class Dataloader(object):
    def __init__(self, dataloader, queue_size, cache_size):
        self.dataqueue = queue.Queue(queue_size)
        self.datalaoder = dataloader
        self.cache_size = cache_size

    def _allocate_tensor(self, x, device=None):
        if device is None:
            device = self.main_device
        else:
            device = torch.device(device)
        if self.dataparallel:
            return x
        else:
            if isinstance(x, list):
                return [i.to(device) for i in x]
            elif isinstance(x, tuple):
                return tuple([i.to(device) for i in x])
            else:
                return x.to(device)

    def feed_thread(self):
        self.nfiles = 0
        for files in self.dataloader:

            # move to GPU or keep in RAM

            tmplist = []
            while self.dataqueue.qsize() > self.cache_size:
                time.sleep(1)
            self.dataqueue.put(tmplist)

    def StartDataCaching(self):
        self.thread = Thread(target=self.feed_thread)
        self.thread.start()

    def get(self):
        return self.dataqueue.get()

    def __len__(self):
        return len(self.datalaoder)

    def __next__(self):
        if self.dataqueue.empty():
            while self.dataqueue.empty():
                time.sleep(1)
        return self.get()
