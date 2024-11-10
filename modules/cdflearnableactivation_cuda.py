import torch
import torch.nn as nn
from collections import defaultdict
import datetime

class CDFLearnableActivation(nn.Module):
    def __init__(self, normalize_by_area=True, device='cpu'):
        super(CDFLearnableActivation, self).__init__()
        self.normalize_by_area = normalize_by_area
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.scale = nn.Parameter(torch.tensor(1.0, device=self.device))
        self.value_counts = defaultdict(int)
        self.cdf_computed = False
        self.cdf_dict = {}
        self.sorted_values = None

    def update_histogram(self, x):
        print(f"[{datetime.datetime.now()}] Updating histogram...")
        x = x.to('cpu')  # Histogram updates are on CPU
        rounded_x = (torch.round(x * 100) / 100).flatten()
        unique, counts = torch.unique(rounded_x, return_counts=True, sorted=True)
        for val, count in zip(unique.tolist(), counts.tolist()):
            self.value_counts[val] += count
        self.cdf_computed = False

    def compute_cdf(self):
        if not self.cdf_computed:
            sorted_values = sorted(self.value_counts.keys())
            frequencies = torch.tensor([self.value_counts[val] for val in sorted_values], dtype=torch.float32, device=self.device)
            cumulative_frequencies = torch.cumsum(frequencies, dim=0)
            cdf = cumulative_frequencies / cumulative_frequencies[-1]
            self.cdf_dict = {sv: cdf[i].item() for i, sv in enumerate(sorted_values)}
            self.sorted_values = torch.tensor(sorted_values, device=self.device)
            self.cdf_computed = True

    def map_values_to_cdf(self, x):
        self.compute_cdf()
        rounded_x = torch.round(x * 100) / 100
        indices = torch.searchsorted(self.sorted_values, rounded_x.view(-1), right=True)
        indices = indices.clamp(0, len(self.sorted_values) - 1)
        cdf_values = torch.tensor([self.cdf_dict[self.sorted_values[idx].item()] for idx in indices], dtype=torch.float32, device=self.device)
        return self.scale * cdf_values.view_as(x)

    def forward(self, x):
        if self.training:
            self.update_histogram(x.detach())
        x = x.to(self.device)
        cdf_values = self.map_values_to_cdf(x)
        return cdf_values

class ModifiedCNN(SimpleCNN):
    def __init__(self, num_classes, device='cpu'):
        super(ModifiedCNN, self).__init__(num_classes)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.custom_activation = CDFLearnableActivation(normalize_by_area=False, device=self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.pool(self.custom_activation(self.conv1(x)))
        x = self.pool(self.custom_activation(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.custom_activation(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ModifiedCNN(num_classes=10, device=device)
