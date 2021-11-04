import torch
from torchvision import models


class LossNet(torch.nn.Module):
    def __init__(self):
        super(LossNet, self).__init__()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        vgg16 = models.vgg16(pretrained=True).to(device)
        self.module_list = list(vgg16.features)
        self.need_layer = [3, 8, 17, 26, 35]

    def forward(self, inputs):
        result = []
        x = self.module_list[0](inputs)
        for i in range(1, len(self.module_list)):
            x = self.module_list[i](x)
            if i in self.need_layer:
                result.append(x)
        return result
