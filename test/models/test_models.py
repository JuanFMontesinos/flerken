import unittest
from flerken.models import UNet, resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet200, vgg_m
import torch


class TestUNet(unittest.TestCase):
    def test_unet_build(self):
        model = UNet([16, 32, 64, 128], 5, verbose=True, useBN=True)
        x = torch.rand(1, 1, 256, 256)
        out = model(x)


class Test3DResNet(unittest.TestCase):
    def test_restnet10(self):
        model = resnet10(num_classes=400)
        x = torch.rand(1, 3, 12, 122, 122)
        out = model(x)

    def test_restnet18(self):
        model = resnet18(pretrained=True, num_classes=400)
        x = torch.rand(1, 3, 12, 122, 122)
        out = model(x)

    def test_restnet34(self):
        model = resnet34(pretrained=True, num_classes=400)
        x = torch.rand(1, 3, 12, 122, 122)
        out = model(x)

    def test_restnet50(self):
        model = resnet50(pretrained=True, num_classes=400)
        x = torch.rand(1, 3, 12, 122, 122)
        out = model(x)

    def test_restnet101(self):
        model = resnet101(pretrained=True, num_classes=400)
        x = torch.rand(1, 3, 12, 122, 122)
        out = model(x)

    def test_restnet152(self):
        model = resnet152(pretrained=True, num_classes=400)
        x = torch.rand(1, 3, 12, 122, 122)
        out = model(x)

    def test_restnet200(self):
        model = resnet200(num_classes=400)
        x = torch.rand(1, 3, 12, 122, 122)
        out = model(x)


class TestVGG(unittest.TestCase):
    def test_vggm(self):
        model = vgg_m(num_classes=400)
        x = torch.rand(1, 3, 12, 122, 122)
