from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from transformers import WavLMConfig, WavLMModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import wavlm_loss


class WavLMTrimmingTests(unittest.TestCase):
    def test_trim_preserves_hidden_state_four_for_both_normalization_modes(self):
        torch.manual_seed(42)
        for stable in (False, True):
            config = WavLMConfig(hidden_size=32, num_hidden_layers=6, num_attention_heads=4,
                intermediate_size=64, conv_dim=(16, 16, 16), conv_stride=(2, 2, 2),
                conv_kernel=(4, 4, 4), num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=4,
                do_stable_layer_norm=stable, layerdrop=0, mask_time_prob=0)
            model = WavLMModel(config).eval()
            wave = torch.randn(1, 1024)
            with torch.no_grad():
                expected = model(wave, output_hidden_states=True).hidden_states[4]
            with patch.object(wavlm_loss.WavLMModel, 'from_pretrained', return_value=model):
                trimmed = wavlm_loss.load_wavlm(trim=True)
            with torch.no_grad():
                actual = trimmed(wave, output_hidden_states=True).hidden_states[4]
            torch.testing.assert_close(actual, expected)
            self.assertEqual(len(trimmed.encoder.layers), 4)
            self.assertTrue(all(not p.requires_grad for p in trimmed.parameters()))


if __name__ == "__main__":
    unittest.main()
