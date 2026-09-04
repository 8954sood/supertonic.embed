# -*- coding: utf-8 -*-
"""The released SupertonicTTS, converted to frozen differentiable PyTorch.

The four released ONNX modules are converted so gradients can flow back to a
style vector. Every converted parameter is frozen: nothing in the model is ever
trained, and only a style vector is handed to an optimizer.

Also here: the prompt, which is a sentence together with the fixed noise latent
it is always synthesized from. Drawing the latent once from a fixed seed and
reusing it means the only thing that changes during a descent is the style.
"""
import os
from dataclasses import dataclass

import numpy as np
import onnxslim
import torch
from onnx2torch import convert

from config import DEVICE, ONNX_DIR, SEED, SPEED, SR, TEXTS, TOTAL_STEP
from style_file import preset_path, read_style
from text_tokenizer import TextTokenizer

MODULES = ("duration_predictor", "text_encoder", "vector_estimator", "vocoder")


def _load_frozen(name, onnx_dir):
    """Slim one released ONNX module, drop it to opset 17, convert, and freeze."""
    slimmed = onnxslim.slim(os.path.join(onnx_dir, f"{name}.onnx"))
    for opset in slimmed.opset_import:
        if opset.domain in ("", "ai.onnx"):
            opset.version = 17
    for node in slimmed.graph.node:
        # onnx2torch cannot convert a Clip with trailing empty inputs
        if node.op_type == "Clip":
            kept = [i for i in node.input if i != ""]
            del node.input[:]
            node.input.extend(kept)
    module = convert(slimmed)
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module.to(DEVICE)


@dataclass
class Prompt:
    """A sentence and the fixed noise latent it is synthesized from."""
    text: str
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
    """The four modules, the tokenizer, and the shapes tying them together."""

    def __init__(self, onnx_dir=None):
        onnx_dir = onnx_dir or ONNX_DIR
        self.tokenize = TextTokenizer(os.path.join(onnx_dir, "unicode_indexer.json"))
        self.duration, self.text_encoder, self.vector_estimator, self.vocoder = (
            _load_frozen(name, onnx_dir) for name in MODULES)
        import json
        with open(os.path.join(onnx_dir, "tts.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        # a latent frame covers `chunk` output samples, and carries `latent_dim`
        self.chunk = cfg["ae"]["base_chunk_size"] * cfg["ttl"]["chunk_compress_factor"]
        self.latent_dim = cfg["ttl"]["latent_dim"] * cfg["ttl"]["chunk_compress_factor"]

    # ----- text -----------------------------------------------------------

    def encode(self, text, lang="en"):
        """-> (ids [1,T], mask [1,1,T]) as tensors on the device."""
        ids, mask = self.tokenize(text, lang)
        return (torch.tensor(ids, dtype=torch.int64, device=DEVICE),
                torch.tensor(mask, dtype=torch.float32, device=DEVICE))

    # ----- the differentiable pass ----------------------------------------

    def forward(self, ids, mask, style_ttl, style_dp, latent, latent_mask,
                total_step=TOTAL_STEP, speed=SPEED):
        """All four modules, differentiable end to end. -> (waveform, duration)

        Any batch size works: each style vector reaches only its own output, so
        a batch of B styles is equivalent to B separate runs.
        """
        b = style_ttl.shape[0]
        duration = self.duration(ids, style_dp, mask) / speed
        text_emb = self.text_encoder(ids, style_ttl, mask)
        latent = latent * latent_mask
        total = torch.full((b,), float(total_step), device=DEVICE)
        for step in range(total_step):
            now = torch.full((b,), float(step), device=DEVICE)
            latent = self.vector_estimator(latent, text_emb, style_ttl,
                                           latent_mask, mask, now, total)
        return self.vocoder(latent), duration

    def predict_seconds(self, prompt, style_dp, speed=SPEED):
        """How long this style says the prompt should take."""
        with torch.no_grad():
            return float((self.duration(prompt.ids, style_dp, prompt.mask)
                          / speed).reshape(()))

    # ----- synthesis ------------------------------------------------------

    def synthesize(self, prompt, style_ttl, style_dp, total_step=TOTAL_STEP,
                   speed=SPEED):
        """The prompt at its fixed latent length. -> 1-D waveform."""
        with torch.no_grad():
            wav, _ = self.forward(prompt.ids, prompt.mask, style_ttl, style_dp,
                                  prompt.latent, prompt.latent_mask, total_step, speed)
        return wav.reshape(-1)

    def synthesize_at_own_length(self, prompt, style_ttl, style_dp, seed=SEED,
                                 total_step=TOTAL_STEP, speed=SPEED):
        """The prompt at the length THIS duration style commands, as inference
        does, so a recovered style_dp is heard at the tempo it asks for."""
        seconds = self.predict_seconds(prompt, style_dp, speed)
        frames = max(4, int(np.ceil(int(seconds * SR) / self.chunk)))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latent = torch.randn(1, self.latent_dim, frames, generator=generator).to(DEVICE)
        latent_mask = torch.ones(1, 1, frames, device=DEVICE)
        with torch.no_grad():
            wav, _ = self.forward(prompt.ids, prompt.mask, style_ttl, style_dp,
                                  latent, latent_mask, total_step, speed)
        return wav.reshape(-1).cpu().numpy(), seconds


def build_prompts(tts, texts=TEXTS, seed=SEED, speed=SPEED, sizing_preset="M1"):
    """One prompt per sentence, each with its own fixed noise latent.

    The latent length comes from one preset's predicted duration, the same for
    every speaker, so the descent compares like with like: only the timbre
    differs between two runs, never the length.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    _, sizing_dp = read_style(preset_path(sizing_preset))

    prompts = []
    for text in texts:
        ids, mask = tts.encode(text)
        with torch.no_grad():
            seconds = float((tts.duration(ids, sizing_dp, mask) / speed).reshape(()))
        frames = int(np.ceil(int(seconds * SR) / tts.chunk))
        prompts.append(Prompt(
            text=text,
            ids=ids,
            mask=mask,
            latent=torch.tensor(np.random.randn(1, tts.latent_dim, frames)
                                .astype(np.float32), device=DEVICE),
            latent_mask=torch.ones(1, 1, frames, device=DEVICE)))
    return prompts
