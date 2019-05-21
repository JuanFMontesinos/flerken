import unittest
from flerken.utils import losses
import torch

class TestContrastiveLoss(unittest.TestCase):
    def setUp(self) -> None:
        self.loss = losses.ContrastiveLoss()
    def test_assertion_input(self):
        pass


