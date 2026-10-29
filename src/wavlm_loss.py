"""The objective the timbre is inverted against: WavLM layer statistics.

A voice is the mean and standard deviation over time of one WavLM-Large layer.
Pooling over time makes the loss independent of what is said, and it is the same
pooling x-vector speaker embeddings use. Layer 4 is where Chiu et al. (2025)
found speaker identity peaks; HuggingFace's hidden_states[4] is that layer.
"""
import librosa
import torch
import torch.nn.functional as F
from transformers import WavLMModel

from config import DEVICE, LAYER, SR, WAVLM_SR

def load_wavlm(trim=False):
    """WavLM-Large, frozen."""
    model = WavLMModel.from_pretrained("microsoft/wavlm-large").eval()
    if trim:
        # hidden_states[LAYER] is before the original encoder's final norm.
        # Truncation keeps the same statistic and avoids unused later blocks.
        if model.config.do_stable_layer_norm:
            model.encoder.layer_norm = torch.nn.Identity()
        model.encoder.layers = model.encoder.layers[:LAYER]
        model.config.num_hidden_layers = LAYER
    model = model.to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def to_wavlm_rate(wav):
    """44.1 kHz to 16 kHz, differentiably, keeping the batch dimension."""
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    n = int(round(wav.shape[-1] * WAVLM_SR / SR))
    return F.interpolate(wav.unsqueeze(1), size=n, mode="linear",
                         align_corners=False).squeeze(1)


def layer_stats(wavlm, wav16):
    """(mean, std) over time of the chosen layer, each [B, D]."""
    h = wavlm(wav16, output_hidden_states=True).hidden_states[LAYER]
    return h.mean(dim=1), h.std(dim=1)


def style_loss(synthesized, target):
    """Mean squared error between two (mean, std) pairs."""
    return (F.mse_loss(synthesized[0], target[0])
            + F.mse_loss(synthesized[1], target[1]))


def target_from_file(wavlm, path):
    """The statistics of a reference recording, at any input sample rate."""
    y, _ = librosa.load(path, sr=WAVLM_SR)
    wav = torch.tensor(y, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        return layer_stats(wavlm, wav)
