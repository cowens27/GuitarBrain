import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset


class ChordDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        self.labels = []
        self.label_map = {'C':0, 'G':1, 'D':2}

        for label in os.listdir(root_dir):
            folder = os.path.join(root_dir, label)
            for file in os.listdir(folder):
                self.data.append(os.path.join(folder, file))
                self.labels.append(self.label_map[label])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        label = self.labels[idx]

        # Load Audio
        y, sr = librosa.load(file_path, sr=22050)

        # Convert into Spectrogram
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        spec = librosa.power_to_db(spec, ref=np.max)

        # Normalize
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)

        # Resize / Pad to fixed size
        spec = spec[:, :128] # Truncate time axis
        if spec.shape[1] < 128:
            pad_width = 128 - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad_width)))

        # Convert to tensor
        spec = torch.tensor(spec).unsqueeze(0).float()

        return spec, label

# Returns spectrogram for a given input
def get_input(path):
    y, sr = librosa.load(path, sr=22050)

    spec = librosa.feature.melspectrogram(y=y,sr=sr, n_mels=128)
    spec = librosa.power_to_db(spec, ref=np.max)

    spec = (spec - spec.mean()) / (spec.std() + 1e-6)

    spec = spec[:, :128]
    if spec.shape[1] < 128:
        pad_width = 128-spec.shape[1]
        spec = np.pad(spec, ((0,0), (0, pad_width)))

    spec = torch.tensor(spec).unsqueeze(0).float()

    return spec