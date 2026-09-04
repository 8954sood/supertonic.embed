# -*- coding: utf-8 -*-
"""The objective the timbre style is inverted against.

A voice is described by the mean and standard deviation, over time, of one
WavLM-Large layer. Collapsing the time axis is what makes the objective
content-independent: the reference and the synthesis may say different words.
It is also the statistics pooling that x-vector speaker embeddings use, which is
why matching it transfers to speaker-verification metrics it never saw.

Layer 4 is the default because Chiu et al. (2025) report peak
speaker-identification accuracy there for this model. Their layer 0 is the CNN
encoder output, which matches HuggingFace's `hidden_states` indexing, so
`hidden_states[4]` is their layer 4.
"""
import os

import torch
import torch.nn.functional as F

from config import DEVICE, LAYER, SR, WAVLM_SR

# Some networks intercept TLS, and the HuggingFace client then refuses to
# download the weights. Verification is disabled for that client only.
os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("CURL_CA_BUNDLE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)
import httpx  # noqa: E402

_verified_client = httpx.Client


class _UnverifiedClient(_verified_client):
    def __init__(self, *args, **kwargs):
        kwargs["verify"] = False
        super().__init__(*args, **kwargs)


httpx.Client = _UnverifiedClient


def load_wavlm():
    """WavLM-Large, frozen. Nothing in it is ever trained."""
    from transformers import WavLMModel
    model = WavLMModel.from_pretrained("microsoft/wavlm-large").to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def to_wavlm_rate(wav):
    """Resample 44.1 kHz to WavLM's 16 kHz, differentiably, keeping the batch."""
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    n = int(round(wav.shape[-1] * WAVLM_SR / SR))
    return F.interpolate(wav.unsqueeze(1), size=n, mode="linear",
                         align_corners=False).squeeze(1)


def layer_stats(wavlm, wav16, layer=LAYER):
    """(mean, standard deviation) over time of one layer. Each is [B, D]."""
    h = wavlm(wav16, output_hidden_states=True).hidden_states[layer]
    return h.mean(dim=1), h.std(dim=1)


def style_loss(synthesized, target):
    """Mean squared error between two (mean, std) pairs."""
    return (F.mse_loss(synthesized[0], target[0])
            + F.mse_loss(synthesized[1], target[1]))


def target_from_file(wavlm, path, layer=LAYER):
    """The statistics of a reference recording, read at any sample rate.

    The objective pools over time, so the recording has to be mostly speech: a
    file that is largely silence gives a target describing the room, not a voice.
    """
    import librosa
    y, _ = librosa.load(path, sr=WAVLM_SR)
    wav = torch.tensor(y, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        return layer_stats(wavlm, wav, layer)
