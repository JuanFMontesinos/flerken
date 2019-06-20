import torch
import torch.nn as nn

__all__=['UNet']

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
    return img[:, :, j:j + w, i:i + h]


def center_crop(img, output_size):
    """This function is prepared to crop tensors provided by dataloader.
    Cited tensors has shape [1,N_maps,H,W]
    """
    _, _, w, h = img.size()
    th, tw = output_size[0], output_size[1]
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)


class ConvolutionalBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_conv=3, kernel_MP=2, stride_conv=1, stride_MP=2, padding=1, bias=True,
                 useBN=False):
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
        self.useBN = useBN
        if self.useBN:
            self.Conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_conv, stride=stride_conv, padding=padding,
                                   bias=bias)
            self.BN1 = nn.BatchNorm2d(dim_out)
            self.ReLu1 = nn.LeakyReLU(0.1)
            self.Conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=kernel_conv, stride=stride_conv, padding=padding,
                                   bias=bias)
            self.BN2 = nn.BatchNorm2d(dim_out)
            self.ReLu2 = nn.LeakyReLU(0.1)
        else:
            self.Conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_conv, stride=stride_conv, padding=padding,
                                   bias=bias)
            self.ReLu1 = nn.ReLU(0)
            self.Conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=kernel_conv, stride=stride_conv, padding=padding,
                                   bias=bias)
            self.ReLu2 = nn.ReLU(0)
        self.MaxPooling = nn.MaxPool2d(kernel_size=kernel_MP, stride=stride_MP, padding=0, dilation=1,
                                       return_indices=False, ceil_mode=False)

    def forward(self, x):
        if self.useBN:
            x = self.Conv1(x)
            x = self.BN1(x)
            x = self.ReLu1(x)
            x = self.Conv2(x)
            x = self.BN2(x)
            to_cat = self.ReLu2(x)
            to_down = self.MaxPooling(to_cat)
        else:
            x = self.Conv1(x)
            x = self.ReLu1(x)
            x = self.Conv2(x)
            to_cat = self.ReLu2(x)
            to_down = self.MaxPooling(to_cat)

        return to_cat, to_down


class AtrousBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_conv=3, kernel_UP=2, stride_conv=1, stride_UP=2, padding=1, bias=True,
                 useBN=False, finalblock=False, printing=False):
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
        if self.useBN:

            self.Conv1 = nn.Conv2d(2 * dim_in, dim_in, kernel_size=kernel_conv, stride=stride_conv, padding=padding,
                                   bias=bias)
            self.BN1 = nn.BatchNorm2d(dim_in)
            self.ReLu1 = nn.LeakyReLU(0.1)
            self.Conv2 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_conv, stride=stride_conv, padding=padding,
                                   bias=bias)
            self.BN2 = nn.BatchNorm2d(dim_in)
            self.ReLu2 = nn.LeakyReLU(0.1)
        else:

            self.Conv1 = nn.Conv2d(2 * dim_in, dim_in, kernel_size=kernel_conv, stride=stride_conv, padding=padding,
                                   bias=bias)
            self.ReLu1 = nn.ReLU(0)
            self.Conv2 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_conv, stride=stride_conv, padding=padding,
                                   bias=bias)
            self.ReLu2 = nn.ReLU(0)

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
            x = self.ReLu1(x)
            x = self.Conv2(x)
            x = self.BN2(x)
            x = self.ReLu2(x)
            if not self.finalblock:
                x = self.AtrousConv(x)
        else:
            to_cat = center_crop(to_cat, x.size()[2:4])
            x = torch.cat((x, to_cat), dim=1)
            x = self.Conv1(x)
            x = self.ReLu1(x)
            x = self.Conv2(x)
            x = self.ReLu2(x)
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

    def __init__(self, dim_in, dim_out, kernel_conv=3, kernel_UP=2, stride_conv=1, stride_UP=2, padding=1, bias=True,
                 useBN=False, ):
        super(TransitionBlock, self).__init__()
        self.useBN = useBN
        if self.useBN:
            self.Conv1 = nn.Conv2d(int(dim_in / 2), dim_in, kernel_size=kernel_conv, stride=stride_conv,
                                   padding=padding, bias=bias)
            self.BN1 = nn.BatchNorm2d(dim_in)
            self.ReLu1 = nn.LeakyReLU(0.1)
            self.Conv2 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_conv, stride=stride_conv, padding=padding,
                                   bias=bias)
            self.BN2 = nn.BatchNorm2d(dim_in)
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

    def forward(self, x):
        if self.useBN:
            x = self.Conv1(x)
            x = self.BN1(x)
            x = self.ReLu1(x)
            x = self.Conv2(x)
            x = self.BN2(x)
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

    def __init__(self, dimensions_vector, K, verbose=False, useBN=False, input_channels=1,
                 activation=None, **kwargs):
        super(UNet, self).__init__()
        self.K = K
        self.printing = verbose

        self.useBN = useBN

        self.input_channels = input_channels
        self.dim = dimensions_vector
        self.vec = range(len(self.dim))
        self.encoder = self.add_encoder(input_channels)
        self.decoder = self.add_decoder(**kwargs)

        self.activation = activation
        self.final_conv = nn.Conv2d(self.dim[0], self.K, kernel_size=1, stride=1, padding=0)
        if self.activation is not None:
            self.final_act = self.activation

    def add_encoder(self, input_channels):
        encoder = []
        for i in range(len(self.dim) - 1):  # There are len(self.dim)-1 downconv blocks
            if self.printing:
                print('Building Downconvolutional Block {} ...OK'.format(i))
            if i == 0:
                """SET 1 IF GRAYSCALE OR 3 IF RGB========================================"""
                encoder.append(ConvolutionalBlock(input_channels, self.dim[i], useBN=self.useBN))
            else:
                encoder.append(ConvolutionalBlock(self.dim[i - 1], self.dim[i], useBN=self.useBN))
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
                    TransitionBlock(self.dim[i], self.dim[i - 1], useBN=self.useBN,
                                    **kwargs))
            elif i == 0:  # Special case for last (top) upconv block
                decoder.append(
                    AtrousBlock(self.dim[i], self.dim[i - 1], finalblock=True, useBN=self.useBN))
            else:
                decoder.append(AtrousBlock(self.dim[i], self.dim[i - 1], useBN=self.useBN))
        decoder = nn.Sequential(*decoder)
        return decoder

    def forward(self, x):
        if self.printing:
            print('UNet input size {0}'.format(x.size()))
        to_cat_vector = []
        for i in range(len(self.dim) - 1):
            if self.printing:
                print('Forward Prop through DownConv block {}'.format(i))
            to_cat, x = self.encoder[i](x)
            to_cat_vector.append(to_cat)
        for i in self.vec:
            if self.printing:
                print('Concatenating and Building  UpConv Block {}'.format(i))
            if i == 0:
                x = self.decoder[i](x)
            else:
                x = self.decoder[i](x, to_cat_vector[-i])
        x = self.final_conv(x)
        if self.activation is not None:
            x = self.final_act(x)
        if self.printing:
            print('UNet Output size {}'.format(x.size()))
        else:
            return x


