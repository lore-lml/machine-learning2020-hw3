import torch
import torch.nn as nn
from torch.autograd import Function

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['AlexDann', 'alexdann']

model_urls = {
    'alexdann': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexDann(nn.Module):

    def __init__(self, num_classes=7):
        super(AlexDann, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.dann_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2),
        )

    def forward(self, x, alpha=None):
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if alpha is None:
            x = self.classifier(x)
        else:
            x = ReverseLayerF.grad_reverse(x, alpha)
            x = self.dann_classifier(x)
        return x


def alexdann(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexDann(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexdann'],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
        for i in [1, 4, 6]:
            model.dann_classifier[i].weight.data = model.classifier[i].weight.data
            model.dann_classifier[i].bias.data = model.classifier[i].bias.data

    return model


class ReverseLayerF(Function):
    # Forwards identity
    # Sends backward reversed gradients
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

    @staticmethod
    def grad_reverse(x, constant):
        return ReverseLayerF.apply(x, constant)

