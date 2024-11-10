import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 120)
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CDFActivation(nn.Module):
    """
        Activation based on cumulative distribution function (CDF) with a trainable scale parameter.
    """
    def __init__(self, normalize_by_area=True):
        """
            Instance initialization method.
            param normalize_by_area (bool): Switch to control normalization by maximum value or area.
        """
        super(CDFActivation, self).__init__()
        self.normalize_by_area = normalize_by_area
        # Initialize a trainable scale parameter
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        original_shape = x.shape
        
        if len(original_shape) == 4:
            batch_size, channels, height, width = original_shape
            x = x.view(batch_size, channels, -1)
        elif len(original_shape) == 2:
            batch_size, features = original_shape
            channels = 1
            x = x.view(batch_size, channels, -1)

        sorted_x, _ = torch.sort(x, dim=2)
        cumulative_sum = torch.cumsum(sorted_x, dim=2)

        if self.normalize_by_area:
            total_sum = cumulative_sum[:, :, -1].view(batch_size, channels, 1)
            cdf = cumulative_sum / total_sum
        else:
            max_height = torch.max(cumulative_sum, dim=2)[0].view(batch_size, channels, 1)
            cdf = cumulative_sum / max_height

        cdf = self.scale * cdf

        if len(original_shape) == 4:
            cdf = cdf.view(batch_size, channels, height, width)
        elif len(original_shape) == 2:
            cdf = cdf.view(batch_size, features)

        return cdf


class ModifiedCNN(SimpleCNN):
    def __init__(self, num_classes):
        super(ModifiedCNN, self).__init__(num_classes)
        self.custom_activation = CDFActivation(normalize_by_area=False)

    def forward(self, x):
        x = self.pool(self.custom_activation(self.conv1(x)))
        x = self.pool(self.custom_activation(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.custom_activation(self.fc1(x))
        x = self.fc2(x)

        return x

