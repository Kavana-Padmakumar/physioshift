"""
PhysioShift - Lightweight 1D-ResNet model
20 Aug 2026 - Member 2 (backfilled) - Phase 3, Day 5
~500K parameter, 6-block 1D-ResNet for physiological signal classification
"""

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """A single 1D residual block: conv -> bn -> relu -> conv -> bn, plus skip connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # If shape changes (channels or stride), project the skip connection to match
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # the "skip connection"
        out = self.relu(out)
        return out


class LightweightResNet1D(nn.Module):
    """
    A lightweight 6-block 1D-ResNet for classifying physiological signals
    (ECG/PPG windows). Targets ~500K parameters.
    """

    def __init__(self, num_classes=5, in_channels=1, base_channels=48):
        super().__init__()

        # Initial stem: bring raw signal up to base_channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        # 6 residual blocks, channels double twice to keep params near 500K
        self.layer1 = ResidualBlock1D(base_channels, base_channels, stride=1)
        self.layer2 = ResidualBlock1D(base_channels, base_channels, stride=1)
        self.layer3 = ResidualBlock1D(base_channels, base_channels * 2, stride=2)
        self.layer4 = ResidualBlock1D(base_channels * 2, base_channels * 2, stride=1)
        self.layer5 = ResidualBlock1D(base_channels * 2, base_channels * 4, stride=2)
        self.layer6 = ResidualBlock1D(base_channels * 4, base_channels * 4, stride=1)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x):
        # x expected shape: (batch, in_channels, sequence_length)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.global_pool(x)       # (batch, channels, 1)
        x = x.squeeze(-1)             # (batch, channels)
        x = self.fc(x)                # (batch, num_classes)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Sanity check: build the model, run a fake input through it, print param count
    model = LightweightResNet1D(num_classes=5, in_channels=1, base_channels=48)

    dummy_input = torch.randn(4, 1, 1000)  # batch=4, 1 channel, 1000 timesteps
    output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total trainable parameters: {count_parameters(model):,}")