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


if __name__ == "__main__":
    from load.load import SpeechLevelDataset
    from torch.utils.data import DataLoader

    annotations_file_path = Path(
        "data",
        "annotations",
        "true_positive.csv",
    )

    speech_level_data_path = Path(
        "data",
        "speech_level_data",
    )

    dataset = SpeechLevelDataset(
        annotations_file_path=annotations_file_path,
        speech_level_data_path=speech_level_data_path,
        waveform_normalize=True,
        mean=-4.1389,
        std=5.3054,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=None,
        shuffle=True,
    )

    data = next(iter(dataloader))

    adapted_frame_based_AST_params = Path(
        "pretrained_params",
        "adapted_frame_based_AST_params.pt",
    )

    model = SentenceLevelEncoder(
        params_file_path=adapted_frame_based_AST_params,
    )

    output = model(
        audio_inputs=data["audio_inputs"],
        audio_attn_masks=data["audio_attn_masks"],
    )

    output.shape
