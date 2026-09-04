"""The released SupertonicTTS as frozen, differentiable PyTorch.

The four ONNX modules are converted so gradients can reach a style vector; every
converted parameter is frozen. A Prompt is a sentence with the fixed noise latent
it is always synthesized from, so the only thing that changes during a descent
is the style.
"""
import json
import os
from dataclasses import dataclass

import numpy as np
import onnxslim
import torch
from onnx2torch import convert

from config import DEVICE, ONNX_DIR, SEED, SPEED, SR, TEXTS, TOTAL_STEP
from style_file import preset_path, read_style
from text_tokenizer import TextTokenizer

SIZING_PRESET = "M1"       # whose predicted duration sets every prompt's latent length


def _load_frozen(name):
    """Slim one released ONNX module, drop it to opset 17, convert, freeze."""
    slimmed = onnxslim.slim(os.path.join(ONNX_DIR, f"{name}.onnx"))
    for opset in slimmed.opset_import:
        if opset.domain in ("", "ai.onnx"):
            opset.version = 17
    for node in slimmed.graph.node:
        if node.op_type == "Clip":           # onnx2torch rejects trailing empty inputs
            kept = [i for i in node.input if i != ""]
            del node.input[:]
            node.input.extend(kept)
    module = convert(slimmed).eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module.to(DEVICE)


@dataclass
class Prompt:
    """A sentence and the fixed noise latent it is synthesized from."""
    ids: torch.Tensor           # [1, T]
    mask: torch.Tensor          # [1, 1, T]
    latent: torch.Tensor        # [1, D, L]
    latent_mask: torch.Tensor   # [1, 1, L]

    def repeat(self, n):
        """The same prompt shaped for a batch of n styles."""
        return (self.ids.expand(n, -1).contiguous(),
                self.mask.expand(n, -1, -1).contiguous(),
                self.latent.expand(n, -1, -1).contiguous(),
                self.latent_mask.expand(n, -1, -1).contiguous())


class FrozenTTS:
    """The four modules, the tokenizer, and the latent geometry tying them together."""

    def __init__(self):
        self.tokenize = TextTokenizer(os.path.join(ONNX_DIR, "unicode_indexer.json"))
        self.duration = _load_frozen("duration_predictor")
        self.text_encoder = _load_frozen("text_encoder")
        self.vector_estimator = _load_frozen("vector_estimator")
        self.vocoder = _load_frozen("vocoder")
        with open(os.path.join(ONNX_DIR, "tts.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        compress = cfg["ttl"]["chunk_compress_factor"]
        self.chunk = cfg["ae"]["base_chunk_size"] * compress     # output samples per latent frame
        self.latent_dim = cfg["ttl"]["latent_dim"] * compress

    def encode(self, text, lang="en"):
        """-> (ids [1,T], mask [1,1,T]) on the device."""
        ids, mask = self.tokenize(text, lang)
        return (torch.tensor(ids, dtype=torch.int64, device=DEVICE),
                torch.tensor(mask, dtype=torch.float32, device=DEVICE))

    def forward(self, ids, mask, style_ttl, style_dp, latent, latent_mask):
        """All four modules, differentiable. -> (waveform, seconds)

        Any batch size: each style reaches only its own output, so a batch of B
        styles equals B separate runs.
        """
        b = style_ttl.shape[0]
        seconds = self.duration(ids, style_dp, mask) / SPEED
        text_emb = self.text_encoder(ids, style_ttl, mask)
        latent = latent * latent_mask
        total = torch.full((b,), float(TOTAL_STEP), device=DEVICE)
        for step in range(TOTAL_STEP):
            now = torch.full((b,), float(step), device=DEVICE)
            latent = self.vector_estimator(latent, text_emb, style_ttl,
                                           latent_mask, mask, now, total)
        return self.vocoder(latent), seconds

    def predict_seconds(self, prompt, style_dp):
        """How long this duration style says the prompt takes."""
        with torch.no_grad():
            return float((self.duration(prompt.ids, style_dp, prompt.mask) / SPEED).reshape(()))

    def synthesize(self, prompt, style_ttl, style_dp):
        """The prompt at its fixed latent length. -> 1-D waveform"""
        with torch.no_grad():
            wav, _ = self.forward(prompt.ids, prompt.mask, style_ttl, style_dp,
                                  prompt.latent, prompt.latent_mask)
        return wav.reshape(-1)

    def synthesize_at_own_length(self, prompt, style_ttl, style_dp):
        """The prompt at the length THIS duration style commands, as inference does.
        -> (waveform as numpy, seconds)"""
        seconds = self.predict_seconds(prompt, style_dp)
        frames = max(4, int(np.ceil(int(seconds * SR) / self.chunk)))
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        latent = torch.randn(1, self.latent_dim, frames, generator=generator).to(DEVICE)
        latent_mask = torch.ones(1, 1, frames, device=DEVICE)
        with torch.no_grad():
            wav, _ = self.forward(prompt.ids, prompt.mask, style_ttl, style_dp,
                                  latent, latent_mask)
        return wav.reshape(-1).cpu().numpy(), seconds


def build_prompts(tts, texts=TEXTS):
    """One prompt per sentence, each with its own fixed latent.

    The latent length comes from one preset's predicted duration, the same for
    every speaker, so two descents differ only in timbre, never in length.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    _, sizing_dp = read_style(preset_path(SIZING_PRESET))
    prompts = []
    for text in texts:
        ids, mask = tts.encode(text)
        with torch.no_grad():
            seconds = float((tts.duration(ids, sizing_dp, mask) / SPEED).reshape(()))
        frames = int(np.ceil(int(seconds * SR) / tts.chunk))
        prompts.append(Prompt(
            ids=ids,
            mask=mask,
            latent=torch.tensor(np.random.randn(1, tts.latent_dim, frames).astype(np.float32),
                                device=DEVICE),
            latent_mask=torch.ones(1, 1, frames, device=DEVICE)))
    return prompts
