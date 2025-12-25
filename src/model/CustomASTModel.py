import math
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import ASTConfig, ASTModel
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import (  # noqa: E501
    ASTAttention,
    ASTEmbeddings,
    ASTEncoder,
    ASTLayer,
    ASTPatchEmbeddings,
    ASTSelfAttention,
)


class CustomASTConfig(ASTConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.f_shape = 128
        self.t_shape = 2
        self.frequency_stride = 128
        self.time_stride = 1


class CustomASTPatchEmbeddings(ASTPatchEmbeddings):
    def __init__(self, config):
        super().__init__(config)

        fshape = config.f_shape
        tshape = config.t_shape
        frequency_stride = config.frequency_stride
        time_stride = config.time_stride

        self.projection = nn.Conv2d(
            1,
            config.hidden_size,
            kernel_size=(fshape, tshape),
            stride=(frequency_stride, time_stride),
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        input_values = input_values.unsqueeze(1)
        input_values = input_values.transpose(2, 3)
        embeddings = self.projection(input_values)

        embeddings = embeddings.flatten(2).transpose(1, 2)

        return embeddings


class CustomASTEmbeddings(ASTEmbeddings):
    def __init__(self, config):
        super().__init__(config)

        self.patch_embeddings = CustomASTPatchEmbeddings(config)

    def get_shape(self, config):
        # see Karpathy's cs231n blog on how to calculate the output dimensions
        # https://cs231n.github.io/convolutional-networks/#conv
        frequency_out_dimension = (
            config.num_mel_bins - config.f_shape
        ) // config.frequency_stride + 1
        time_out_dimension = (
            config.max_length - config.t_shape
        ) // config.time_stride + 1

        return frequency_out_dimension, time_out_dimension


class CustomASTSelfAttention(ASTSelfAttention):
    def __init__(self, config):
        super().__init__(config)

        self.attn_map = []
        self.attn_grad = []

    def forward(
        self,
        hidden_states,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        get_attention_weights: bool = False,
        register_hook: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention
        # scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply attention mask
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        def save_attn_gradients(attn_gradients):
            print("AST: suceessfully registered hook!")
            if self.attn_grad != []:
                print("AST: attn_grad is not empty!")
            self.attn_grad.append(attn_gradients.detach())

        # register hook to save attention gradients
        if register_hook:
            attention_probs.register_hook(save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Save attention map as an attribute
        if get_attention_weights:
            if self.attn_map != []:
                print("AST: attn_map is not empty!")
            self.attn_map.append(attention_probs.detach())
            print("AST: suceessfully saved attention weights!")

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs


class CustomASTAttention(ASTAttention):
    def __init__(self, config):
        super().__init__(config)
        self.attention = CustomASTSelfAttention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        # Add Register hook argument to save attention map and gradients
        get_attention_weights: bool = False,
        register_hook: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(
            hidden_states,
            head_mask,
            output_attentions,
            # Add Register hook argument to save attention weights gradients
            get_attention_weights=get_attention_weights,
            register_hook=register_hook,
            attention_mask=attention_mask,
        )

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class CustomASTLayer(ASTLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = CustomASTAttention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        # Add Register hook argument to save gradients
        get_attention_weights: bool = False,
        register_hook: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(
                hidden_states
            ),  # in AST, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
            get_attention_weights=get_attention_weights,
            register_hook=register_hook,
            attention_mask=attention_mask,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in AST, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class CustomASTEncoder(ASTEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList(
            [CustomASTLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        get_attention_weights: bool = False,
        register_hook: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and i > 9 and self.training:
                layer_outputs = checkpoint(
                    layer_module,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                    get_attention_weights,
                    register_hook,
                    attention_mask,
                )
                # print("Encoder: checkpointing layer {}".format(i))

            else:
                layer_outputs = layer_module(
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                    get_attention_weights=get_attention_weights,
                    register_hook=register_hook,
                    attention_mask=attention_mask,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class CustomASTModel(ASTModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = CustomASTEmbeddings(config)
        self.encoder = CustomASTEncoder(config)

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        get_attention_weights: bool = False,
        register_hook: bool = False,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_values is None:
            raise ValueError("You have to specify input_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads
        # x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(input_values)

        extended_attention_mask = self.extend_attention_mask(attention_mask)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            get_attention_weights=get_attention_weights,
            register_hook=register_hook,
            attention_mask=extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        pooled_output = (sequence_output[:, 0] + sequence_output[:, 1]) / 2

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def extend_attention_mask(
        self,
        attention_mask: torch.Tensor,
        dtype=torch.float32,
    ) -> torch.Tensor:
        """Makes broadcastable attention.

        and causal masks so that future and masked tokens are ignored.
        A function modified from HuggingFace Transformers Bert model.
        """
        if attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]

            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for attention_mask (shape {attention_mask.shape})"
            )
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked
        # positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
            dtype
        ).min

        return extended_attention_mask


def create_encoder():
    encoder = CustomASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    return encoder


if __name__ == "__main__":
    configuration = CustomASTConfig()

    ast = CustomASTModel(
        config=configuration,
    )

    adapted_frame_based_AST_params = Path(
        "pretrained_params",
        "adapted_frame_based_AST_params.pt",
    )

    ast.load_state_dict(
        torch.load(
            adapted_frame_based_AST_params,
            # map_location=torch.device("cpu"),
        )
    )


# %%
