import unittest
from flerken.models import UNet, resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet200, vgg_m
import torch


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


class TestUNet(unittest.TestCase):
    def setUp(self) -> None:
        self.x = torch.rand(2, 1, 256, 256)
        self.c = torch.zeros(2, 10).float()

    def test_conditioned_cunet_bnon(self):
        model = UNet([16, 32, 64, 128, 256, 512], 1, [10, 'eld'], useBN=True, verbose=True)
        out = model(self.x, self.c)

    def test_conditioned_unet_dropout(self):
        model = UNet([16, 32, 64, 128], 1, [10, 'eld'], useBN=True, verbose=True, dropout=0.5)
        out = model(self.x, self.c)

    def test_conditioned_unet_bn_momentum(self):
        model = UNet([16, 32, 64, 128], 1, [10, 'eld'], useBN=True, verbose=True, bn_momentum=0.5)
        out = model(self.x, self.c)

    def test_conditioned_cunet_bnoff(self):
        with self.assertRaises(ValueError):
            model = UNet([16, 32, 64, 128, 256, 512], 1, [10, 'eld'], useBN=False, verbose=True)

    def test_conditioned_unet_bnon(self):
        model = UNet([16, 32, 64, 128, 256, 512], 1, None, useBN=True, verbose=True)
        out = model(self.x, self.c)

    def test_dimensions_vector_assertion(self):
        with self.assertRaises(AssertionError):
            model = UNet('8,12,14', 1, None, useBN=False, verbose=True)
        with self.assertRaises(AssertionError):
            model = UNet([7, 14, 28], 1, None, useBN=False, verbose=True)
        with self.assertRaises(ValueError):
            model = UNet([12, 24, 44], 1, None, useBN=False, verbose=True)

    def test_input_channel_assertion(self):
        with self.assertRaises(AssertionError):
            model = UNet([16, 32], 7, None, input_channels='jamon', useBN=False, verbose=True)
        with self.assertRaises(AssertionError):
            model = UNet([16, 32], 7, None, input_channels=-5, useBN=False, verbose=True)

    def test_K_assertion(self):
        with self.assertRaises(AssertionError):
            model = UNet([16, 32], 'pizza', None, useBN=False, verbose=True)
        with self.assertRaises(AssertionError):
            model = UNet([16, 32], -1, None, useBN=False, verbose=True)

    def test_dropout_assertion(self):
        with self.assertRaises(AssertionError):
            model = UNet([16, 32], 1, None, useBN=False, verbose=True, dropout=7)
        with self.assertRaises(AssertionError):
            model = UNet([16, 32], 1, None, useBN=False, verbose=True, dropout='pizza')

    def test_activation(self):
        model = UNet([16, 32, 64, 128], 1, [10, 'eld'], activation=torch.nn.Sigmoid(), useBN=True, verbose=True)
        out = model(self.x, self.c)
        self.assertFalse((out < 0).any().item())
        self.assertFalse((out > 1).any().item())

    def test_unet_bnmomentum(self):
        MOMENTUM = 0.6
        model = UNet([16, 32, 64, 128, 256, 512], 1, None, useBN=True, verbose=True, bn_momentum=MOMENTUM)
        for i in range(len(model.encoder._modules)):
            assert model.encoder._modules[str(i)].BN1.momentum == MOMENTUM
            assert model.encoder._modules[str(i)].BN2.momentum == MOMENTUM
        for i in range(len(model.decoder._modules)):
            assert model.decoder._modules[str(i)].BN1.momentum == MOMENTUM
            assert model.decoder._modules[str(i)].BN2.momentum == MOMENTUM

    def test_unet_bnmomentum_assertion(self):
        MOMENTUM = 'jamon'

        with self.assertRaises(AssertionError):
            model = UNet([16, 32, 64], 1, None, useBN=True, verbose=True, bn_momentum=MOMENTUM)
        with self.assertRaises(AssertionError):
            model = UNet([16, 32, 64], 1, None, useBN=True, verbose=True, bn_momentum=14)
        with self.assertRaises(AssertionError):
            model = UNet([16, 32, 64], 1, None, useBN=True, verbose=True, bn_momentum=-1)
