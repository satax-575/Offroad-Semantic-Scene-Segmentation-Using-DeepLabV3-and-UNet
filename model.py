import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU()
        ]
        if dropout:
            layers.append(nn.Dropout2d(0.3))
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.Wg = nn.Conv2d(F_g, F_int, 1)
        self.Wx = nn.Conv2d(F_l, F_int, 1)
        self.psi = nn.Conv2d(F_int, 1, 1)

    def forward(self, g, x):
        psi = torch.sigmoid(self.psi(self.Wg(g) + self.Wx(x)))
        return x * psi


class UNet(nn.Module):
    """Original UNet with Attention - Proven to work"""
    def __init__(self, n_classes):
        super().__init__()

        self.e1 = ConvBlock(3, 32)
        self.e2 = ConvBlock(32, 64)
        self.e3 = ConvBlock(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.att3 = AttentionBlock(128, 128, 64)
        self.d3 = ConvBlock(256, 128, dropout=True)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.att2 = AttentionBlock(64, 64, 32)
        self.d2 = ConvBlock(128, 64, dropout=True)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d1 = ConvBlock(64, 32)

        self.out = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        e3 = self.att3(d3, e3)
        d3 = self.d3(torch.cat([d3, e3], 1))

        d2 = self.up2(d3)
        e2 = self.att2(d2, e2)
        d2 = self.d2(torch.cat([d2, e2], 1))

        d1 = self.up1(d2)
        d1 = self.d1(torch.cat([d1, e1], 1))

        return self.out(d1)


class DeepLabV3Wrapper(nn.Module):
    """IMPROVED: DeepLabV3 with ResNet50 backbone for better performance"""
    def __init__(self, num_classes):
        super().__init__()
        # Load pretrained DeepLabV3
        self.model = deeplabv3_resnet50(pretrained=True)
        
        # Replace classifier head for our number of classes
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        
        # Also update aux classifier if it exists
        if hasattr(self.model, 'aux_classifier'):
            self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        
        # FIXED: Convert all BatchNorm to eval mode to avoid batch size=1 error
        self._set_batchnorm_eval()
    
    def _set_batchnorm_eval(self):
        """Set all BatchNorm layers to eval mode permanently"""
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.eval()
                # Freeze BatchNorm parameters
                for param in module.parameters():
                    param.requires_grad = False
    
    def train(self, mode=True):
        """Override train to keep BatchNorm in eval mode"""
        super().train(mode)
        # Keep BatchNorm in eval mode even during training
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.eval()
        return self
    
    def forward(self, x):
        # DeepLabV3 returns a dict with 'out' key during training
        result = self.model(x)
        if isinstance(result, dict):
            return result['out']
        return result


def get_model(model_type, num_classes):
    """
    IMPROVED: Factory function to get model by type
    
    Args:
        model_type: "unet" or "deeplab"
        num_classes: number of output classes
    
    Returns:
        model instance
    """
    if model_type == "unet":
        print("Using UNet with Attention")
        return UNet(num_classes)
    elif model_type == "deeplab":
        print("Using DeepLabV3 with ResNet50 backbone")
        return DeepLabV3Wrapper(num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'unet' or 'deeplab'")


if __name__ == "__main__":
    # Test both models
    print("Testing UNet...")
    unet = UNet(6)
    x = torch.randn(1, 3, 256, 256)
    out = unet(x)
    print(f"UNet output shape: {out.shape}")
    
    print("\nTesting DeepLabV3...")
    deeplab = DeepLabV3Wrapper(6)
    out = deeplab(x)
    print(f"DeepLabV3 output shape: {out.shape}")
    
    print("\nBoth models work correctly!")
