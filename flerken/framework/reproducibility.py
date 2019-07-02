import torch
import numpy as np
import random


class ReproducibilityModule(object):
    def __init__(self, hard: bool):
        self.hard = hard

    @staticmethod
    def get_seed():
        torch_seed = torch.initial_seed()
        numpy_seed = np.random.get_state()
        random_seed = random.getstate()
        return torch_seed, numpy_seed, random_seed

    @staticmethod
    def _save_seed(self, torch_seed, numpy_seed, random_seed):
        pick = [torch_seed, numpy_seed, random_seed]
