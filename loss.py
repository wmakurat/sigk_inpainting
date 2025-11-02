import torch
import torch.nn as nn
from torchvision import models


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)

        self.encoders = nn.ModuleList([
            nn.Sequential(*vgg16.features[:5]),
            nn.Sequential(*vgg16.features[5:10]),
            nn.Sequential(*vgg16.features[10:17]),
        ])

        for enc in self.encoders:
            for param in enc.parameters():
                param.requires_grad = False

    def forward(self, image):
        results = []
        x = image
        for enc in self.encoders:
            x = enc(x)
            results.append(x)
        return results
    

def gram_matrix(feature):
    B, C, H, W = feature.shape
    Fm = feature.flatten(2)
    G = torch.bmm(Fm, Fm.transpose(1, 2))

    G = G / (H * W)
    G = G / (C * C)

    return G


def total_variation_loss(image):
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class InpaintingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = VGG16FeatureExtractor()

    def forward(self, input, mask, output, gt):
        output_comp = mask * input + (1 - mask) * output

        loss_hole = self.l1((1 - mask) * output, (1 - mask) * gt)
        loss_valid = self.l1(mask * output, mask * gt)

        feature_output_comp = self.extractor(output_comp)
        feature_output = self.extractor(output)
        feature_gt = self.extractor(gt)

        loss_prerceptual = 0.0
        for i in range(3):
            loss_prerceptual += self.l1(feature_output[i], feature_gt[i])
            loss_prerceptual += self.l1(feature_output_comp[i], feature_gt[i])

        loss_style = 0.0
        for i in range(3):
            loss_style += self.l1(gram_matrix(feature_output[i]),
                                          gram_matrix(feature_gt[i]))
            loss_style += self.l1(gram_matrix(feature_output_comp[i]),
                                          gram_matrix(feature_gt[i]))

        loss_tv = total_variation_loss(output_comp)

        return loss_valid + 6*loss_hole + 0.05*loss_prerceptual + 120*(loss_style) + 0.1*loss_tv
