import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PerceptualLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        vgg_features = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.vgg = nn.Sequential(*list(vgg_features.children())[:36]).to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.loss_fn = F.mse_loss

    def forward(self, generated_img, target_img):
        features_gen = self.vgg(generated_img)
        features_target = self.vgg(target_img)
        return self.loss_fn(features_gen, features_target)


class HybridLoss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_perceptual=0.1, device='cuda'):
    # lambda_l1: L1损失的权重
    # lambda_perceptual: 感知损失的权重
    # device: 设备
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.l1_loss = nn.L1Loss().to(device)
        self.perceptual_loss = PerceptualLoss(device=device)

    def forward(self, generated_img, target_img):
        l1_val = self.l1_loss(generated_img, target_img)
        perceptual_val = self.perceptual_loss(generated_img, target_img)
        total = self.lambda_l1 * l1_val + self.lambda_perceptual * perceptual_val
        return total, l1_val, perceptual_val
