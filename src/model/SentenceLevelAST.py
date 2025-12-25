from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from model.CustomASTModel import CustomASTConfig, CustomASTModel

# from azureml.fsspec import AzureMachineLearningFileSystem
from src.load.load import SegmentLevelDataset


class SentenceLevelAST(nn.Module):
    def __init__(
        self,
        reinit_encoders_n_layers=0,
        params_file_path: Path = Path(
            "data",
            "pretrained_params",
            "adapted_frame_based_AST_params.pt",
        ),
    ):
        super().__init__()

        configuration = CustomASTConfig()

        self.encoder = CustomASTModel(
            config=configuration,
        )

        self.out_proj = nn.Linear(768, 2)

        self.encoder.load_state_dict(
            torch.load(
                params_file_path,
                map_location=torch.device("cpu"),
            )
        )

        self.reinit_n_layers = reinit_encoders_n_layers

        if self.reinit_n_layers > 0:
            self.reinitialize_layers()

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

        output = self.out_proj(outputs[:, 0])

        return output

    def reinitialize_layers(self):
        # Reinitialize the LayerNorm layer of the AST sentence level encoder
        self.encoder.layernorm.weight.data.fill_(1.0)
        self.encoder.layernorm.bias.data.zero_()

        # Reinitialize the last n attention layers of the sentence level encoders
        for n in range(self.reinit_n_layers):
            self.encoder.encoder.layer[-(n + 1)].apply(self._initialize_weights_biases)

    def _initialize_weights_biases(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0,
                std=self.encoder.config.initializer_range,
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


if __name__ == "__main__":
    segment_level_dataset = SegmentLevelDataset(
        waveform_normalize=True,
    )

    dataloader = DataLoader(
        segment_level_dataset,
        batch_size=2,
        shuffle=True,
    )

    data = next(iter(dataloader))

    audio_inputs = data["audio_inputs"]
    audio_attn_masks = data["audio_attn_masks"]

    model = SentenceLevelAST(reinit_encoders_n_layers=2)

    output = model(
        audio_inputs,
        audio_attn_masks,
    )
