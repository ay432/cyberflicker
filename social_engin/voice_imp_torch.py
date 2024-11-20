import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

def wav2spectrum(filename):
    waveform, sr = torchaudio.load(filename)
    # Convert to spectrogram (log scale for better visualization)
    spectrogram = torchaudio.transforms.MelSpectrogram()(waveform)
    log_spectrogram = torch.log1p(spectrogram)
    return log_spectrogram, sr

def spectrum2wav(spectrogram, sr, output_filename):
    # Convert spectrogram back to waveform
    inverse_transform = torchaudio.transforms.InverseMelScale(n_mels=128, sample_rate=sr)
    waveform = inverse_transform(spectrogram.exp())  # Reverse log1p
    torchaudio.save(output_filename, waveform, sr)

class RandomCNN(nn.Module):
    def __init__(self):
        super(RandomCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 128 * 128, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def compute_content_loss(content, generated):
    return torch.mean((content - generated) ** 2)

def compute_style_loss(style, generated):
    return torch.mean((style - generated) ** 2)

import time
import math

# Define paths to the content and style audio files
CONTENT_FILENAME = "content_audio.wav"
STYLE_FILENAME = "style_audio.wav"

# Load content and style audio files
a_content, sr = wav2spectrum(CONTENT_FILENAME)
a_style, _ = wav2spectrum(STYLE_FILENAME)

# Convert to torch tensors
a_content_torch = a_content.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
a_style_torch = a_style.unsqueeze(0).unsqueeze(0)

# Initialize the model
model = RandomCNN()
model.eval()

# Optimization setup
a_G_var = torch.randn_like(a_content_torch, requires_grad=True)
optimizer = torch.optim.Adam([a_G_var], lr=0.002)

# Style and content weight parameters
style_param = 1
content_param = 1e2

# Training loop
num_epochs = 2000
print_every = 100

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()
for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad()

    # Forward pass: generate audio
    a_G = model(a_G_var)

    # Calculate losses
    content_loss = content_param * compute_content_loss(a_content_torch, a_G)
    style_loss = style_param * compute_style_loss(a_style_torch, a_G)

    # Total loss
    loss = content_loss + style_loss

    # Backpropagation
    loss.backward()
    optimizer.step()

    if epoch % print_every == 0:
        print(f"{epoch} {epoch / num_epochs * 100:.2f}% {timeSince(start)} "
              f"content_loss:{content_loss.item():.4f} style_loss:{style_loss.item():.4f} total_loss:{loss.item():.4f}")

# Save the generated spectrogram and convert it back to audio
gen_spectrum = a_G_var.cpu().data.numpy().squeeze()
spectrum2wav(torch.tensor(gen_spectrum), sr, "generated_audio.wav")
