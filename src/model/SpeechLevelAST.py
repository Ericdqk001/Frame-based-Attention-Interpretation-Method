from pathlib import Path

import torch

# from ParticipantLevelEncoder import ParticipantLevelEncoder
from torch import nn

from model.SentenceLevelEncoder import SentenceLevelEncoder
from model.SpeechLevelEncoder import SpeechLevelEncoder


class SpeechLevelAST(nn.Module):
    def __init__(
        self,
        reinit_encoders_n_layers=0,
        num_layers_pl_encoder=4,
        params_file_path=Path(
            "data",
            "pretrained_params",
            "adapted_frame_based_AST_params.pt",
        ),
    ):
        super().__init__()

        self.sentence_level_block = nn.DataParallel(
            SentenceLevelEncoder(
                params_file_path=params_file_path,
            ),
        )

        self.speech_level_block = SpeechLevelEncoder(num_layers=num_layers_pl_encoder)

        self.reinit_n_layers = reinit_encoders_n_layers

        if self.reinit_n_layers > 0:
            self.reinitialize_layers()

    def forward(
        self,
        audio_inputs: torch.tensor = None,
        audio_attn_masks: torch.tensor = None,
        get_attention_weights=False,
        average_attn_weights=False,
        register_hook=False,
    ):
        """A AST model that operates at a speech level.

        check the paper at http://arxiv.org/abs/2309.13476 for details.


        Args:
            audio_inputs (torch.tensor, optional): sentence-level audio data batched
            together. Defaults to None.
            audio_attn_masks (torch.tensor, optional): masks for padding patches.
            Defaults to None.
            get_attention_weights (bool, optional): set True to register attention
            scores. Defaults to False.
            average_attn_weights (bool, optional): set false because we average across
            heads taking into account gradients. Defaults to False.
            register_hook (bool, optional): set True to register gradients with respect
            to attention scores. Defaults to False.

        Returns:
            tensor: a tensor of size 2 (binary classification)
        """
        sentence_representation = self.sentence_level_block(
            audio_inputs=audio_inputs,
            audio_attn_masks=audio_attn_masks,
            get_attention_weights=get_attention_weights,
            register_hook=register_hook,
        )

        # To have a batch size of 1 and a sequence of sentence representations
        sentence_representation = sentence_representation.permute(1, 0, 2)

        output = self.speech_level_block(
            sentence_representation,
            get_attention_weights=get_attention_weights,
            average_attn_weights=average_attn_weights,
            register_hook=register_hook,
        )

        return output

    def reinitialize_layers(self):
        # Reinitialize the LayerNorm layer of the AST sentence level encoder
        self.sentence_level_block.module.encoder.layernorm.weight.data.fill_(1.0)
        self.sentence_level_block.module.encoder.layernorm.bias.data.zero_()

        # Reinitialize the last n attention layers of the sentence level encoders
        for n in range(self.reinit_n_layers):
            self.sentence_level_block.module.encoder.encoder.layer[-(n + 1)].apply(
                self._initialize_weights_biases
            )

    def _initialize_weights_biases(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0,
                std=self.sentence_level_block.module.encoder.config.initializer_range,
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
