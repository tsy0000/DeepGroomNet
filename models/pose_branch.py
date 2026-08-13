import torch
import torch.nn as nn

class PoseBranch(nn.Module):

    def __init__(self, in_channels, num_classes, heatmap_size=64, hidden=128, dropout=0.4):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.conv = nn.Sequential(nn.Conv3d(in_channels, hidden, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool3d(kernel_size=(1, 2, 2)), nn.Conv3d(hidden, hidden // 2, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool3d(kernel_size=(1, 2, 2)), nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(1), nn.Dropout(p=dropout), nn.Linear(hidden // 2, num_classes))
        for m in self.conv:
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, heatmaps):
        x = heatmaps.permute(0, 2, 1, 3, 4)
        return self.conv(x)
