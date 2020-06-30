import math
import torch
import torch.nn as nn
from torch.autograd import Function
from tqdm import tqdm
import copy
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'pytorch': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
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

        self.discriminator = nn.Sequential(
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
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if alpha is None:
            x = self.classifier(x)
        else:
            x = ReverseLayerF.grad_reverse(x, alpha)
            x = self.discriminator(x)
        return x


def alexnet(pretrained=True, progress=True, num_classes=7, **kwargs):

    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[src],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
        model.classifier[6] = nn.Linear(4096, num_classes)
        for i in [1, 4]:
            model.discriminator[i].weight.data = copy.deepcopy(model.classifier[i].weight.data)
            model.discriminator[i].bias.data = copy.deepcopy(model.classifier[i].bias.data)
    else:
        model.classifier[6] = nn.Linear(4096, num_classes)

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


def _alpha_scheduling(i, epoch, min_len, nepochs):
    p = float(i + epoch * min_len) / nepochs / min_len
    o = 10
    alpha = 2. / (1. + math.exp(-o * p)) - 1
    # print 'lamda: %.4f' % lamda
    return alpha


def train_no_dann(model, dataloader, optimizer, criterion, current_step, device='cuda'):
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


def test_target(model, dataloader, device='cuda'):
    model.eval()
    running_corrects = 0
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs.data, 1)
        running_corrects += torch.sum(preds == labels.data).data.item()

    return running_corrects


def dann_train_src_target(model, src_dataloader, tgt_dataloader, optimizer, criterion,
                          current_step, current_epoch, max_epoch, alpha='dynamic', device='cuda'):
    cum_loss_class = .0
    cum_loss_domain = .0
    len_data_loader = min(len(src_dataloader), len(tgt_dataloader))
    dynamic = True if alpha == 'dynamic' else False

    for i, (src_iter, tgt_iter) in enumerate(zip(src_dataloader, tgt_dataloader)):
        src_img, src_labels = src_iter
        tgt_img, tgt_labels = tgt_iter
        src_img = src_img.to(device)
        src_labels = src_labels.to(device)

        src_fake_labels = torch.zeros(len(src_labels), dtype=torch.int64).to(device)
        tgt_img = tgt_img.to(device)
        tgt_fake_labels = torch.ones(len(tgt_labels), dtype=torch.int64).to(device)

        model.train()
        optimizer.zero_grad()

        if dynamic:
            alpha = _alpha_scheduling(i, current_epoch, len_data_loader, max_epoch)

        # CLASSIFIER
        class_outputs = model(src_img)
        class_loss = criterion(class_outputs, src_labels)
        cum_loss_class += class_loss.item()
        class_loss.backward()

        # DISCRIMINATOR SRC
        domain_src_outputs = model(src_img, alpha=alpha)
        loss_src_d = criterion(domain_src_outputs, src_fake_labels)
        # DISCRIMINATOR TGT
        domain_tgt_outputs = model(tgt_img, alpha=alpha)
        loss_tgt_d = criterion(domain_tgt_outputs, tgt_fake_labels)
        domain_loss = loss_src_d + loss_tgt_d
        domain_loss.backward()
        cum_loss_domain += domain_loss.item()

        if current_step % 10 == 0:
            print(f"Step {current_step}\nClass Loss {class_loss.item()}, Domain Loss {domain_loss.item()}")

        optimizer.step()
        current_step += 1

    return (cum_loss_class / len_data_loader), (cum_loss_domain / len_data_loader), current_step
