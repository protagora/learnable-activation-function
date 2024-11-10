import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

import datetime

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

# class CDFLearnableActivation(nn.Module):
#     def __init__(self, normalize_by_area=True):
#         super(CDFLearnableActivation, self).__init__()
#         self.normalize_by_area = normalize_by_area
#         self.scale = nn.Parameter(torch.tensor(1.0))
#         self.value_counts = defaultdict(int)

#     def update_histogram(self, x):
#         print(f"[{datetime.datetime.now()}] Updating histogram...")
#         # Round to 2 decimal places
#         rounded_x = torch.round(x * 100) / 100
#         unique, counts = torch.unique(rounded_x, return_counts=True, sorted=True)
#         for val, count in zip(unique.tolist(), counts.tolist()):
#             self.value_counts[val] += count

#     def compute_cdf(self):
#         sorted_values = sorted(self.value_counts.keys())
#         frequencies = [self.value_counts[val] for val in sorted_values]
#         cumulative_frequencies = torch.cumsum(torch.tensor(frequencies, dtype=torch.float32), dim=0)
#         cdf = cumulative_frequencies / cumulative_frequencies[-1]
#         cdf_dict = dict(zip(sorted_values, cdf))
#         return cdf_dict, sorted_values

#     def map_values_to_cdf(self, x, cdf_dict, sorted_values):
#         rounded_x = torch.round(x * 100) / 100
#         if self.training:
#             # During training, allow new values to widen the histogram
#             min_val, max_val = min(sorted_values), max(sorted_values)
#             clamped_x = torch.clamp(rounded_x, min_val, max_val)
#         else:
#             # During inference, clamp values to ensure they are always in range
#             clamped_x = rounded_x
#         return torch.tensor([cdf_dict.get(val.item(), 0.0) for val in clamped_x.flatten()], dtype=torch.float32).view_as(x)

#     def forward(self, x):
#         if self.training:
#             self.update_histogram(x.detach())
#         cdf_dict, sorted_values = self.compute_cdf()
#         cdf_values = self.map_values_to_cdf(x, cdf_dict, sorted_values)
#         cdf_values = self.scale * cdf_values
#         return cdf_values

class CDFLearnableActivation(nn.Module):
    def __init__(self, normalize_by_area=True):
        super(CDFLearnableActivation, self).__init__()
        self.normalize_by_area = normalize_by_area
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.value_counts = defaultdict(int)
        self.cdf_computed = False
        self.cdf_dict = {}
        self.sorted_values = []

    def update_histogram(self, x):
        print(f"[{datetime.datetime.now()}] Updating histogram...")
        rounded_x = (torch.round(x * 100) / 100).flatten()
        unique, counts = torch.unique(rounded_x, return_counts=True, sorted=True)
        for val, count in zip(unique.tolist(), counts.tolist()):
            self.value_counts[val] += count
        self.cdf_computed = False  # Invalidate the CDF

    def compute_cdf(self):
        if not self.cdf_computed:
            sorted_values = sorted(self.value_counts.keys())
            frequencies = [self.value_counts[val] for val in sorted_values]
            cumulative_frequencies = torch.cumsum(torch.tensor(frequencies, dtype=torch.float32), dim=0)
            cdf = cumulative_frequencies / cumulative_frequencies[-1]
            self.cdf_dict = dict(zip(sorted_values, cdf))
            self.sorted_values = torch.tensor(sorted_values)
            self.cdf_computed = True

    def map_values_to_cdf(self, x):
        self.compute_cdf()  # Ensure CDF is up-to-date
        rounded_x = torch.round(x * 100) / 100
        indices = torch.searchsorted(self.sorted_values, rounded_x.view(-1), right=True)
        indices = indices.clamp(0, len(self.sorted_values) - 1)
        cdf_values = torch.tensor([self.cdf_dict[self.sorted_values[idx].item()] for idx in indices], dtype=torch.float32)
        return self.scale * cdf_values.view_as(x)

    def forward(self, x):
        if self.training:
            self.update_histogram(x.detach())
        cdf_values = self.map_values_to_cdf(x)
        return cdf_values


# Usage in a simple CNN as an activation function
class ModifiedCNN(SimpleCNN):
    def __init__(self, num_classes):
        super(ModifiedCNN, self).__init__(num_classes)
        self.custom_activation = CDFLearnableActivation(normalize_by_area=False)

    def forward(self, x):
        x = self.pool(self.custom_activation(self.conv1(x)))
        x = self.pool(self.custom_activation(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.custom_activation(self.fc1(x))
        x = self.fc2(x)
        return x
