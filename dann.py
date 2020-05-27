import torch
import torch.nn as nn
from torch.autograd import Function
from tqdm import tqdm
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['AlexDann', 'alexdann']

model_urls = {
    'alexdann': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexDann(nn.Module):

    def __init__(self, num_classes=1000):
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

        self.domain_classifier = nn.Sequential(
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
            x = self.domain_classifier(x)
        return x


def alexdann(pretrained=True, progress=True, num_classes=7, **kwargs):
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
        model.classifier[6] = nn.Linear(4096, num_classes)
        model.domain_classifier[6] = nn.Linear(4096, num_classes)
        for i in [1, 4, 6]:
            model.domain_classifier[i].weight.data = model.classifier[i].weight.data
            model.domain_classifier[i].bias.data = model.classifier[i].bias.data

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


def train_src(model, dataloader, optimizer, criterion, current_step, device='cuda'):
    cumulative_loss =.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        model.train()

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        cumulative_loss += loss.item()

        if current_step % 10 == 0:
            print('Step {}, Loss_train {}'.format(current_step, loss.item()))

        loss.backward()
        optimizer.step()
        current_step += 1

    return cumulative_loss, current_step


def test_target(model, dataloader, criterion, device='cuda'):
    model.eval()
    running_corrects = 0
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs.data, 1)
        running_corrects += torch.sum(preds == labels.data).data.item()

    return running_corrects


def dann_train_src_target(model, src_dataloader, tgt_dataloader, optimizer, criterion, current_step, alpha=0.1, device='cuda'):
    cum_loss_src_y = .0
    cum_loss_src_d = .0
    cum_loss_tgt_d = .0

    for (src_img, src_labels), (tgt_img, tgt_labels) in zip(src_dataloader, tgt_dataloader):
        src_img = src_img.to(device)
        src_labels = src_labels.to(device)
        src_fake_labels = torch.zeros(len(src_labels), 1).to(device)
        tgt_img = tgt_img.to(device)
        tgt_fake_labels = torch.ones(len(tgt_labels), 1).to(device)

        model.train()
        optimizer.zero_grad()

        #TRAIN ON SRC CLASSIFIER BRANCH
        outputs = model(src_img)
        loss_src_y = criterion(outputs, src_labels)
        cum_loss_src_y += loss_src_y.item()
        loss_src_y.backward()

        # TRAIN ON SRC DISCRIMINATOR BRANCH
        outputs = model(src_img, alpha=alpha)
        loss_src_d = criterion(outputs, src_fake_labels)
        cum_loss_src_d += loss_src_d.item()
        loss_src_d.backward()

        # TRAIN ON TARGET DISCRIMINATOR BRANCH
        outputs = model(tgt_img, alpha=alpha)
        loss_tgt_d = criterion(outputs, tgt_fake_labels)
        cum_loss_tgt_d += loss_tgt_d.item()
        loss_tgt_d.backward()

        if current_step % 10 == 0:
            print(f"Step {current_step}\nLoss SRC Gy {loss_src_y.item()}, Loss SRC Gd {loss_src_d.item()}, Loss SRC Gy {loss_tgt_d.item()}")

        optimizer.step()
        current_step += 1

    return cum_loss_src_y, cum_loss_src_d, cum_loss_tgt_d, current_step

