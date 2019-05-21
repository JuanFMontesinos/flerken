import unittest
from flerken.utils import losses
import torch


class TestContrastiveLoss(unittest.TestCase):
    def setUp(self) -> None:
        self.loss = losses.ContrastiveLoss()

    def test_assertions(self):
        x = torch.tensor(7.)
        y = torch.tensor([[1.]])
        with self.assertRaises(IndexError):
            self.loss([x, x],y)
        # assert y_type.size()[0] == x0_type.size()[0]
        # IndexError: tuple index  out of range


        x = torch.tensor([7.])
        y = torch.tensor([1.])
        with self.assertRaises(AssertionError):
            self.loss([x, x],y)
        # assert y_type.dim() == 2
        # AssertionError


        x = torch.tensor([7.]).unsqueeze(1)
        y = torch.tensor([1.])
        with self.assertRaises(AssertionError):
            self.loss([x, x],y)
        # assert y_type.dim() == 2
        # AssertionError

    def test_forward_y1(self):
        a = torch.tensor([7.])
        b = torch.tensor([6.5])
        y = torch.tensor([[1.]])

        loss = self.loss([a,b],y)
        loss = loss.item()
        result = 0.5**2
        self.assertEqual(result,loss)

    def test_forward_y0(self):
        # margin > Dist > 0
        a = torch.tensor([7.])
        b = torch.tensor([6.5])
        y = torch.tensor([[0.]])

        loss = self.loss([a,b],y)
        loss = loss.item()
        dist = 0.5
        md = 1-dist
        result = md**2
        self.assertEqual(result,loss)


        # Dist > margin
        a = torch.tensor([7.])
        b = torch.tensor([0.])
        y = torch.tensor([[0.]])

        loss = self.loss([a,b],y)
        loss = loss.item()

        self.assertEqual(0,loss)

        # Dist =0
        a = torch.tensor([7.])
        b = torch.tensor([7.])
        y = torch.tensor([[0.]])

        loss = self.loss([a,b],y)
        loss = loss.item()

        self.assertEqual(1,loss)

