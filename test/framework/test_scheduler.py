import unittest
from flerken.framework.classitems import Scheduler, NullScheduler
import torch
from torch.optim.lr_scheduler import *


class TestScheduler(unittest.TestCase):
    def setUp(self):
        a = torch.nn.Linear(10, 20)
        self.opt = torch.optim.SGD(a.parameters(), lr=1)

    def tearDown(self):
        sch = Scheduler(self.sch)
        sd = sch.state_dict()
        sch.load_state_dict(sd)
        str(sch)

    def test_lambdalr(self):
        lambda1 = lambda epoch: epoch // 30
        self.sch = LambdaLR(self.opt, lr_lambda=lambda1)

    def test_steplr(self):
        self.sch = StepLR(self.opt, step_size=30, gamma=0.1)

    def test_exponentiallr(self):
        self.sch = ExponentialLR(self.opt, gamma=0.1)

    def test_multisteplr(self):
        self.sch = MultiStepLR(self.opt, milestones=[30, 80], gamma=0.1)

    def test_cosineannealinglr(self):
        self.sch = CosineAnnealingLR(self.opt, 10, eta_min=0, last_epoch=-1)

    def test_plateau(self):
        self.sch = ReduceLROnPlateau(self.opt, 'min')

    def cycliclr(self):
        self.sch = CyclicLR(optimizer)
