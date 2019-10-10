import torch
import torch.nn as nn
from numbers import Number
from warnings import warn

__all__ = ['UNet']


def isnumber(x):
    return isinstance(x, Number)


def crop(img, i, j, h, w):
    """Crop the given Image.
    Args:
        img Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    return img[:, :, i:i + h, j:j + w]


def center_crop(img, output_size):
    """This function is prepared to crop tensors provided by dataloader.
    Cited tensors has shape [1,N_maps,H,W]
    """
    _, _, h, w = img.size()
    th, tw = output_size[0], output_size[1]
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)


class ConvolutionalBlock(nn.Module):
    def __init__(self, dim_in, dim_out, film, kernel_conv=3, kernel_MP=2, stride_conv=1, stride_MP=2, padding=1,
                 bias=True, dropout=False,
                 useBN=False, bn_momentum=0.1, **kwargs):
        super(ConvolutionalBlock, self).__init__()
        """Defines a (down)convolutional  block
        Args:
            dim_in: int dimension of feature maps of block input.
            dim_out: int dimension of feature maps of block output.
            kernel_conv: int or tuple kernel size for convolutions
            kernel_MP: int or tuple kernel size for Max Pooling
            stride_conv: int or tuple stride for convolutions
            stride_MP: int or tuple stride for Max Pooling
            padding: padding for convolutions
            bias: bool Set bias or not
            useBN: Use batch normalization

        Forward:
            Returns:
                to_cat: output previous to Max Pooling for skip connections
                to_down: Max Pooling output to be used as input for next block
        """
        assert isinstance(dropout, Number)
        self.film = film
        self.useBN = useBN
        self.dropout = dropout
        if self.useBN:
            self.Conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_conv, stride=stride_conv, padding=padding,
                                   bias=bias)
            self.BN1 = nn.BatchNorm2d(dim_out, momentum=bn_momentum)
            self.ReLu1 = nn.LeakyReLU(0.1)
            self.Conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=kernel_conv, stride=stride_conv, padding=padding,
                                   bias=bias)
            self.BN2 = nn.BatchNorm2d(dim_out, momentum=bn_momentum)
            self.ReLu2 = nn.LeakyReLU(0.1)
        else:
            self.Conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_conv, stride=stride_conv, padding=padding,
                                   bias=bias)
            self.ReLu1 = nn.ReLU(0)
            self.Conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=kernel_conv, stride=stride_conv, padding=padding,
                                   bias=bias)
            self.ReLu2 = nn.ReLU(0)
        if self.dropout:
            self.DO1 = nn.Dropout2d(self.dropout)
            self.DO2 = nn.Dropout2d(self.dropout)
        self.MaxPooling = nn.MaxPool2d(kernel_size=kernel_MP, stride=stride_MP, padding=0, dilation=1,
                                       return_indices=False, ceil_mode=False)

        if isnumber(self.film):
            self.scale = nn.Linear(self.film, dim_out)
            self.bias = nn.Linear(self.film, dim_out)

    def forward(self, *args):
        if isnumber(self.film):
            x, c = args
        else:
            x = args[0]

        if self.useBN:
            x = self.Conv1(x)
            x = self.BN1(x)
            x = self.ReLu1(x)
            if self.dropout:
                x = self.DO1(x)
            x = self.Conv2(x)
            x = self.BN2(x)
            if isnumber(self.film):
                x = self.scale(c).unsqueeze(2).unsqueeze(2) * x + self.bias(c).unsqueeze(2).unsqueeze(2)
            to_cat = self.ReLu2(x)
            if self.dropout:
                to_cat = self.DO2(to_cat)
            to_down = self.MaxPooling(to_cat)
        else:
            x = self.Conv1(x)
            x = self.ReLu1(x)
            if self.dropout:
                x = self.DO1(x)
            x = self.Conv2(x)
            to_cat = self.ReLu2(x)
            if self.dropout:
                to_cat = self.DO2(to_cat)
            to_down = self.MaxPooling(to_cat)

        return to_cat, to_down


class AtrousBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_conv=3, kernel_UP=2, stride_conv=1, stride_UP=2, padding=1, bias=True,
                 useBN=False, finalblock=False, printing=False, bn_momentum=0.1, dropout=False, **kwargs):
        """Defines a upconvolutional  block
        Args:
            dim_in: int dimension of feature maps of block input.
            dim_out: int dimension of feature maps of block output.
            kernel_conv: int or tuple kernel size for convolutions
            kernel_MP: int or tuple kernel size for Max Pooling
            stride_conv: int or tuple stride for convolutions
            stride_MP: int or tuple stride for Max Pooling
            padding: padding for convolutions
            bias: bool Set bias or not
            useBN: Use batch normalization
            finalblock: bool Set true if it's the last upconv block not to do upconvolution.
        Forward:
            Input:
                x: previous block input.
                to_cat: skip connection input.
            Returns:
                x: block output
        """
        super(AtrousBlock, self).__init__()
        self.useBN = useBN
        self.finalblock = finalblock
        self.printing = printing
        self.dropout = dropout
        assert isinstance(dropout, Number)
        if self.useBN:

            self.Conv1 = nn.Conv2d(2 * dim_in, dim_in, kernel_size=kernel_conv, stride=stride_conv, padding=padding,
                                   bias=bias)
            self.BN1 = nn.BatchNorm2d(dim_in, momentum=bn_momentum)
            self.ReLu1 = nn.LeakyReLU(0.1)
            self.Conv2 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_conv, stride=stride_conv, padding=padding,
                                   bias=bias)
            self.BN2 = nn.BatchNorm2d(dim_in, momentum=bn_momentum)
            self.ReLu2 = nn.LeakyReLU(0.1)
        else:

            self.Conv1 = nn.Conv2d(2 * dim_in, dim_in, kernel_size=kernel_conv, stride=stride_conv, padding=padding,
                                   bias=bias)
            self.ReLu1 = nn.ReLU(0)
            self.Conv2 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_conv, stride=stride_conv, padding=padding,
                                   bias=bias)
            self.ReLu2 = nn.ReLU(0)
        if self.dropout:
            self.DO1 = nn.Dropout2d(self.dropout)
            self.DO2 = nn.Dropout2d(self.dropout)
        if not finalblock:
            self.AtrousConv = nn.ConvTranspose2d(dim_in, dim_out, kernel_size=kernel_UP, stride=stride_UP, padding=0,
                                                 dilation=1)

    def forward(self, x, to_cat=None):
        if self.printing:
            print('Incoming variable from previous Upconv Block: {}'.format(x.size()))

        if self.useBN:
            to_cat = center_crop(to_cat, x.size()[2:4])
            x = torch.cat((x, to_cat), dim=1)
            x = self.Conv1(x)
            x = self.BN1(x)
            if self.dropout and not self.finalblock:
                x = self.DO1(x)
            x = self.ReLu1(x)
            x = self.Conv2(x)
            x = self.BN2(x)
            x = self.ReLu2(x)
            if self.dropout and not self.finalblock:
                x = self.DO2(x)
            if not self.finalblock:
                x = self.AtrousConv(x)
        else:
            to_cat = center_crop(to_cat, x.size()[2:4])
            x = torch.cat((x, to_cat), dim=1)
            x = self.Conv1(x)
            x = self.ReLu1(x)
            if self.dropout:
                x = self.DO1(x)
            x = self.Conv2(x)
            x = self.ReLu2(x)
            if self.dropout:
                x = self.DO2(x)
            if not self.finalblock:
                x = self.AtrousConv(x)
        return x


class TransitionBlock(nn.Module):
    """Specific class for lowest block. Change values carefully.
        Args:
            dim_in: int dimension of feature maps of block input.
            dim_out: int dimension of feature maps of block output.
            kernel_conv: int or tuple kernel size for convolutions
            kernel_MP: int or tuple kernel size for Max Pooling
            stride_conv: int or tuple stride for convolutions
            stride_MP: int or tuple stride for Max Pooling
            padding: padding for convolutions
            bias: bool Set bias or not
            useBN: Use batch normalization
        Forward:
            Input:
                x: previous block input.
            Returns:
                x: block output
    """

    def __init__(self, dim_in, dim_out, film, kernel_conv=3, kernel_UP=2, stride_conv=1, stride_UP=2, padding=1,
                 bias=True,
                 useBN=False, bn_momentum=0.1, **kwargs):
        super(TransitionBlock, self).__init__()
        self.film = film
        self.useBN = useBN
        if self.useBN:
            self.Conv1 = nn.Conv2d(int(dim_in / 2), dim_in, kernel_size=kernel_conv, stride=stride_conv,
                                   padding=padding, bias=bias)
            self.BN1 = nn.BatchNorm2d(dim_in, momentum=bn_momentum)
            self.ReLu1 = nn.LeakyReLU(0.1)
            self.Conv2 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_conv, stride=stride_conv, padding=padding,
                                   bias=bias)
            self.BN2 = nn.BatchNorm2d(dim_in, momentum=bn_momentum)
            self.ReLu2 = nn.LeakyReLU(0.1)
        else:
            self.Conv1 = nn.Conv2d(int(dim_in / 2), dim_in, kernel_size=kernel_conv, stride=stride_conv,
                                   padding=padding, bias=bias)
            self.ReLu1 = nn.ReLU(0)
            self.Conv2 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_conv, stride=stride_conv, padding=padding,
                                   bias=bias)
            self.ReLu2 = nn.ReLU(0)
        self.AtrousConv = nn.ConvTranspose2d(dim_in, dim_out, kernel_size=kernel_UP, stride=stride_UP, padding=0,
                                             dilation=1)
        if isnumber(self.film):
            self.scale = nn.Linear(self.film, dim_in)
            self.bias = nn.Linear(self.film, dim_in)

    def forward(self, *args):
        if isnumber(self.film):
            x, c = args
        else:
            x = args[0]
        if self.useBN:
            x = self.Conv1(x)
            x = self.BN1(x)
            x = self.ReLu1(x)
            x = self.Conv2(x)
            x = self.BN2(x)
            if isnumber(self.film):
                x = self.scale(c).unsqueeze(2).unsqueeze(2) * x + self.bias(c).unsqueeze(2).unsqueeze(2)
            x = self.ReLu2(x)
        else:
            x = self.Conv1(x)
            x = self.ReLu1(x)
            x = self.Conv2(x)
            x = self.ReLu2(x)
        to_up = self.AtrousConv(x)

        return to_up


class UNet(nn.Module):
    """It's recommended to be very careful  while managing vectors, since they are inverted to
    set top blocks as block 0. Notice there are N upconv blocks and N-1 downconv blocks as bottom block
    is considered as upconvblock.

    C-U-Net based on this paper https://arxiv.org/pdf/1904.05979.pdf
    """
    """
    Example:
    model = UNet([64,128,256,512,1024,2048,4096],K,useBN=True,input_channels=1)
    
        K(int) : Amount of outgoing channels 
        useBN (bool): Whether to use or not batch normalization
        input_channels (int): Amount of input channels
        dimension_vector (tuple/list): Its length defines amount of block. Elements define amount of filters per block

        
                          
    """

    # TODO Use bilinear interpolation in addition to upconvolutions

    def __init__(self, dimensions_vector, K, film, verbose=False, useBN=False, input_channels=1,
                 activation=None, **kwargs):
        super(UNet, self).__init__()
        self.K = K
        self.printing = verbose
        self.film = film
        self.useBN = useBN

        self.input_channels = input_channels
        self.dim = dimensions_vector
        self.init_assertion(**kwargs)

        self.vec = range(len(self.dim))
        self.encoder = self.add_encoder(input_channels, **kwargs)
        self.decoder = self.add_decoder(**kwargs)

        self.activation = activation
        self.final_conv = nn.Conv2d(self.dim[0], self.K, kernel_size=1, stride=1, padding=0)
        if self.activation is not None:
            self.final_act = self.activation

    def init_assertion(self, **kwargs):
        assert isinstance(self.dim, (tuple, list))
        for x in self.dim:
            assert x % 2 == 0
        if list(map(lambda x: x / self.dim[0], self.dim)) != list(map(lambda x: 2 ** x, range(0, len(self.dim)))):
            raise ValueError('Dimension vector must double their channels sequentially ej. [16,32,64,128,...]')
        assert isinstance(self.input_channels, int)
        assert self.input_channels > 0
        assert isinstance(self.K, int)
        assert self.K > 0
        if kwargs.get('dropout') is not None:
            dropout = kwargs['dropout']
            assert isinstance(dropout, Number)
            assert dropout >= 0
            assert dropout <= 1
        if kwargs.get('bn_momentum') is not None:
            bn_momentum = kwargs['bn_momentum']
            assert isinstance(bn_momentum, Number)
            assert bn_momentum >= 0
            assert bn_momentum <= 1
        if isnumber(self.film) and self.useBN == False:
            raise ValueError(
                'Conditioned U-Net enabled but batch normalization disabled. C-UNet only available with BN on.' \
                ' Note: from a Python perspective, booleans are integers, thus numbers')

    def add_encoder(self, input_channels, **kwargs):
        encoder = []
        for i in range(len(self.dim) - 1):  # There are len(self.dim)-1 downconv blocks
            if self.printing:
                print('Building Downconvolutional Block {} ...OK'.format(i))
            if i == 0:
                """SET 1 IF GRAYSCALE OR 3 IF RGB========================================"""
                encoder.append(ConvolutionalBlock(input_channels, self.dim[i], self.film, useBN=self.useBN, **kwargs))
            else:
                encoder.append(ConvolutionalBlock(self.dim[i - 1], self.dim[i], self.film, useBN=self.useBN, **kwargs))
        encoder = nn.Sequential(*encoder)
        return encoder

    def add_decoder(self, **kwargs):
        decoder = []
        for i in self.vec[::-1]:  # [::-1] inverts the order to set top layer as layer 0 and to order
            # layers from the bottom to above according to  flow of information.
            if self.printing:
                print('Building Upconvolutional Block {}...OK'.format(i))
            if i == max(self.vec):  # Special condition for lowest block
                decoder.append(
                    TransitionBlock(self.dim[i], self.dim[i - 1], self.film, useBN=self.useBN, **kwargs))
            elif i == 0:  # Special case for last (top) upconv block
                decoder.append(
                    AtrousBlock(self.dim[i], self.dim[i - 1], finalblock=True, useBN=self.useBN, **kwargs))
            else:
                decoder.append(AtrousBlock(self.dim[i], self.dim[i - 1], useBN=self.useBN, **kwargs))
        decoder = nn.Sequential(*decoder)
        return decoder

    def forward(self, *args):
        if isnumber(self.film):
            x, c = args
        else:
            x = args[0]
        if self.printing:
            print('UNet input size {0}'.format(x.size()))
        to_cat_vector = []
        for i in range(len(self.dim) - 1):
            if self.printing:
                print('Forward Prop through DownConv block {}'.format(i))
            if isnumber(self.film):

                to_cat, x = self.encoder[i](x, c)
            else:
                to_cat, x = self.encoder[i](x)
            to_cat_vector.append(to_cat)
        for i in self.vec:
            if self.printing:
                print('Concatenating and Building  UpConv Block {}'.format(i))
            if i == 0:
                if isnumber(self.film):
                    x = self.decoder[i](x, c)
                else:
                    x = self.decoder[i](x)
            else:
                x = self.decoder[i](x, to_cat_vector[-i])
        x = self.final_conv(x)
        if self.activation is not None:
            x = self.final_act(x)
        if self.printing:
            print('UNet Output size {}'.format(x.size()))

        return x
