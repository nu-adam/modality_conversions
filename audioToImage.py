import torchaudio
import torchaudio.transforms as T
import torch
import torchvision.transforms as transforms

def preprocess_audio(audio_path, input_size=(224, 224)):
    """
    Preprocesses an audio file into a 3-channel spectrogram image tensor.
    
    Args:
    - audio_path (str): Path to the audio file.
    - input_size (tuple): Desired output size (Height, Width).
    
    Returns:
    - torch.Tensor: Preprocessed spectrogram tensor with 3 channels (like an image).
    """
    SAMPLE_RATE = 16000  # Target sample rate
    DURATION = 3  # Audio clip duration in seconds
    N_MELS = 128  # Mel Spectrogram bands
    TARGET_LENGTH = SAMPLE_RATE * DURATION

    waveform, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLE_RATE)

    if waveform.shape[1] > TARGET_LENGTH:
        waveform = waveform[:, :TARGET_LENGTH]
    else:
        padding = TARGET_LENGTH - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    transform = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=N_MELS)
    mel_spectrogram = transform(waveform)
    mel_spectrogram_db = T.AmplitudeToDB()(mel_spectrogram)
    
    preprocess_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    mel_spectrogram_resized = torch.nn.functional.interpolate(
        mel_spectrogram_db.unsqueeze(0), size=input_size, mode='bilinear'
    ).squeeze(0)

    mel_spectrogram_rgb = mel_spectrogram_resized.repeat(3, 1, 1)

    final_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    spectrogram_tensor = final_transform(mel_spectrogram_rgb)
    
    return spectrogram_tensor
audio_tensor = preprocess_audio('03-01-05-01-01-01-01.wav')
print(audio_tensor.shape)

import matplotlib.pyplot as plt
import torch

def visualize_audio_tensor(audio_tensor):
    """
    Visualizes a preprocessed 3-channel spectrogram tensor as an image.
    
    Args:
    - audio_tensor (torch.Tensor): Tensor of shape (3, 224, 224)
    """
    if audio_tensor.shape[0] != 3:
        raise ValueError("Input tensor must have 3 channels (like an RGB image).")
    
    # Detach and move to CPU if necessary
    audio_tensor = audio_tensor.cpu().detach()
    
    # Normalize tensor to [0, 1] for visualization
    audio_tensor = (audio_tensor - audio_tensor.min()) / (audio_tensor.max() - audio_tensor.min())
    
    # Convert to (H, W, C) for Matplotlib
    audio_image = audio_tensor.permute(1, 2, 0).numpy()
    
    # # Plot the image
    # plt.figure(figsize=(6, 6))
    plt.imshow(audio_image)
    plt.axis('off')
    plt.savefig("spectrogram.png")
    plt.show()
visualize_audio_tensor(audio_tensor)
