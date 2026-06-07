import torch.nn as nn


class Classifier(nn.Module):
    """MLP for tabular lesion features.

    Improvements over the original 20->128->64->32->1 net:
      - configurable depth/width
      - BatchNorm for stable training on heterogeneous feature scales
      - Dropout for regularization (the dataset is small once reduced to features)
    """

    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)
