from torch import nn

__all__ = ['vgg_f', 'vgg_m', 'vgg_s']

"""
Return of the Devil in the Details:Delving Deep into Convolutional Nets
https://arxiv.org/pdf/1405.3531.pdf
"""
class LRN(nn.Module):
    """
    Local Response Normalisatio(LRN) by jiecaoyu
    see: https://github.com/jiecaoyu/pytorch_imagenet/blob/master/networks/model_list/alexnet.py
    """

    def __init__(self, local_size=1, alpha=1E-4, beta=0.75, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()

        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        self.alpha = alpha
        self.beta = beta

        if self.ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(
                kernel_size=(local_size, 1, 1),
                stride=1,
                padding=(int((local_size - 1.0) / 2), 0, 0)
            )
        else:
            self.average = nn.AvgPool2d(
                kernel_size=local_size,
                stride=1,
                padding=int((local_size - 1.0) / 2)
            )

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(2.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(2.0).pow(self.beta)
        x = x.div(div)
        return x


def make_feature(in_channels, out_channels, kernel_size=3, conv_stride=1, conv_pad=0, use_lrn=True, pool_size=2):
    layers = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=conv_stride,
            padding=conv_pad
        ),
        nn.ReLU(True),
    ]

    if use_lrn:
        layers.append(LRN())
    if pool_size > 0:
        layers.append(nn.MaxPool2d(pool_size))

    return layers


class VGG(nn.Module):

    def __init__(self, cfg, num_classes=1000, num_channels=1):
        super(VGG, self).__init__()

        out1, out2, out3, out4 = cfg['filter_size']

        self.features = nn.Sequential(
            *make_feature(num_channels, out1, **cfg['conv1']),
            *make_feature(out1, out2, **cfg['conv2']),
            *make_feature(out2, out3, conv_pad=1, use_lrn=False, pool_size=0),
            *make_feature(out3, out3, conv_pad=1, use_lrn=False, pool_size=0),
            *make_feature(out3, out3, conv_pad=1, use_lrn=False, **cfg.get('conv5', {}))
        )

        self.classifier = nn.Sequential(
            nn.Linear(out4, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def vgg_f(**kwargs):
    return VGG({
        'filter_size': [64, 256, 256, 256 * 6 * 6],
        'conv1': {
            'kernel_size': 11,
            'conv_stride': 4
        },
        'conv2': {
            'kernel_size': 5,
            'conv_stride': 1,
            'conv_pad': 2
        },
    }, **kwargs)


def vgg_m(**kwargs):
    return VGG({
        #        'filter_size': [96, 256, 512, 512*6*6],
        'filter_size': [96, 256, 512, 10752],
        'conv1': {
            'kernel_size': 7,
            'conv_stride': 2
        },
        'conv2': {
            'kernel_size': 5,
            'conv_stride': 2,
            'conv_pad': 1
        },
    }, **kwargs)


def vgg_s(**kwargs):
    return VGG({
        'filter_size': [96, 256, 512, 512 * 5 * 5],
        'conv1': {
            'kernel_size': 7,
            'conv_stride': 2,
            'pool_size': 3
        },
        'conv2': {
            'kernel_size': 5,
            'conv_stride': 1,
            'conv_pad': 1,
            'use_lrn': False
        },
        'conv5': {
            'pool_size': 3
        }
    }, **kwargs)
