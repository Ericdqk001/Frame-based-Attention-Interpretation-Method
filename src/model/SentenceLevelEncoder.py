from pathlib import Path

import torch
from torch import nn

from model.CustomASTModel import CustomASTConfig, CustomASTModel


class SentenceLevelEncoder(nn.Module):
    def __init__(
        self,
        params_file_path: Path,
    ):
        super().__init__()

        configuration = CustomASTConfig()

        self.encoder = CustomASTModel(
            config=configuration,
        )

        self.encoder.load_state_dict(
            torch.load(
                params_file_path,
                # map_location=torch.device("cpu"),
            )
        )

    def forward(
        self,
        audio_inputs,
        audio_attn_masks,
        get_attention_weights: bool = False,
        register_hook: bool = False,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_values=audio_inputs,
            attention_mask=audio_attn_masks,
            get_attention_weights=get_attention_weights,
            register_hook=register_hook,
        )["last_hidden_state"]

        output = outputs[:, 0].unsqueeze(1)

        return output
