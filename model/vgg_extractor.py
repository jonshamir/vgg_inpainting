import torch
import torch.nn as nn
import torchvision.models as models

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class vgg19_conv(nn.Module):
    def __init__(self):
        super(vgg19_conv, self).__init__()
        basic_model = models.vgg19(pretrained=True)
        self.layer_1 = self.make_layers(basic_model,0,1)
        self.layer_2 = self.make_layers(basic_model,1,6)
        self.layer_3 = self.make_layers(basic_model,6,11)
        self.layer_4 = self.make_layers(basic_model,11,20)
        self.layer_5 = self.make_layers(basic_model,20,29)
        self.layers = [self.layer_1, self.layer_2, self.layer_3, self.layer_4, self.layer_5]

    def make_layers(self, basic_model, start_layer, end_layer):
        layer = []
        features = next(basic_model.children())
        original_layer_number = 0
        for module in features.children():
            if original_layer_number >= start_layer and original_layer_number < end_layer:
                layer += [module]
            original_layer_number += 1
        return nn.Sequential(*layer)

class vgg19_relu(nn.Module):
    def __init__(self):
        super(vgg19_relu, self).__init__()
        basic_model = models.vgg19(pretrained=True)
        self.layer_1 = self.make_layers(basic_model,0,2)
        self.layer_2 = self.make_layers(basic_model,2,7)
        self.layer_3 = self.make_layers(basic_model,7,12)
        self.layer_4 = self.make_layers(basic_model,12,21)
        self.layer_5 = self.make_layers(basic_model,21,30)
        self.layers = [self.layer_1, self.layer_2, self.layer_3, self.layer_4, self.layer_5]

    def make_layers(self, basic_model, start_layer, end_layer):
        layer = []
        features = next(basic_model.children())
        original_layer_number = 0
        for module in features.children():
            if original_layer_number >= start_layer and original_layer_number < end_layer:
                layer += [module]
            original_layer_number += 1
        return nn.Sequential(*layer)

vgg_conv = vgg19_conv().to(DEVICE)
vgg_relu = vgg19_relu().to(DEVICE)

def get_VGG_features(batch, relu=False, layer=5):
    if relu: vgg = vgg_relu
    else: vgg = vgg_conv

    out = batch.to(DEVICE)
    for i in range(layer):
        out = vgg.layers[i].forward(out)
    return out


def get_all_VGG_features(batch, relu=False, layer=5):
    if relu: vgg = vgg_relu
    else: vgg = vgg_conv
    feats = []

    out = batch.to(DEVICE)
    for i in range(layer):
        feats.append(out)
        out = vgg.layers[i].forward(out)
    return feats
