import torch


class _Loss(torch.nn.Module):
    def __init__(self, size_average=True, reduce=True):
        super(_Loss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce


class ContrastiveLoss(_Loss):
    def __init__(self, margin=1, size_average=True, reduce=False, weight=None):
        super(ContrastiveLoss, self).__init__(size_average=size_average, reduce=reduce)
        self.register_buffer('weight', weight)
        self.margin = margin

    def input_assertion(self, inputs):
        assert len(inputs) == 3
        y_type, x0_type, x1_type = inputs
        assert x0_type.size() == x1_type.size()
        assert y_type.size()[0] == x0_type.size()[0]
        assert x1_type.size()[0] > 0

        assert y_type.dim() == 2

    def forward(self, X, y):
        x0 = X[0]
        x1 = X[1]
        B = x0.size()[0]
        self.input_assertion((y, x0, x1))
        dist_n = (x0 - x1).view(B, -1).norm(dim=1, )
        loss = y * dist_n.pow(2) + (1 - y) * (self.margin - dist_n).clamp(0, self.margin).pow(2)
        if self.weight is not None:
            loss = loss * self.weight
        if self.reduce:
            return loss
        else:
            if self.size_average:
                return torch.mean(loss)
            else:
                return torch.sum(loss)


class SI_SDR(_Loss):
    """
    SDR - half-baked or well done? 
    https://arxiv.org/abs/1811.02508
    """
    """
    Args:
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size
            "nbatch".
        size_average (bool, optional): By default, the losses are averaged
            over observations for each minibatch. However, if the field
            size_average is set to ``False``, the losses are instead summed for
            each minibatch. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed over
            observations for each minibatch depending on size_average. When reduce
            is False, returns a loss per input/target element instead and ignores
            size_average. Default: True
    """

    def __init__(self, size_average=True, reduce=False, weight=None):
        super(SI_SDR, self).__init__(size_average=size_average, reduce=reduce)
        self.register_buffer('weight', weight)

    def forward(self, s, s_hat):
        assert s.dim() > 1
        assert s_hat.shape == s.shape
        s = s.view(-1, s.shape[-1])
        s_hat = s_hat.view(-1, s_hat.shape[-1])
        dot = torch.bmm(s.unsqueeze(1), s_hat.unsqueeze(2))[:, 0, 0]
        norm = torch.norm(s, 2, 1, True)[:, 0].pow(2)
        coef = dot / norm
        coef = coef.unsqueeze(1)
        s = coef * s
        loss = torch.norm(s, 2, 1, True)[:, 0] / torch.norm(s - s_hat, 2, 1, True)[:, 0]
        loss = 10 * loss.pow(2).log10()
        if self.weight is not None:
            loss = loss * self.weight
        if self.reduce:
            return loss
        else:
            if self.size_average:
                return torch.mean(loss)
            else:
                return torch.sum(loss)
