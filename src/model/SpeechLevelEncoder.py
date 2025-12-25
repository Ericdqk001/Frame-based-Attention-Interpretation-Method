from typing import Callable, Optional, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from model.CustomDecoderLayer import CustomMultiheadAttention


class SpeechLevelEncoder(nn.Module):
    def __init__(
        self,
        dimensionality=768,
        num_layers=4,
        num_heads=6,
    ):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dimensionality))

        self.position_embeddings = nn.Embedding(128, dimensionality)

        self.layers = nn.ModuleList(
            [
                CustomTransformerEncoderLayer(
                    d_model=dimensionality,
                    nhead=num_heads,
                    # Set input and output tensors as the shape of (batch, seq, feature)
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.out_proj = nn.Linear(dimensionality, 2)  # 2 for binary classification

    def forward(
        self,
        decoder_outputs,
        # attention_mask,
        get_attention_weights=False,
        average_attn_weights=False,
        register_hook=False,
    ):
        # Add cls token to the beginning of the sequence
        batch_size = decoder_outputs.size(0)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        embeddings = torch.cat((cls_tokens, decoder_outputs), dim=1)

        # Add position embeddings
        seq_length = embeddings.size(1)

        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=embeddings.device
        )

        position_embeddings = self.position_embeddings(position_ids)

        embeddings = embeddings + position_embeddings

        for layer in self.layers:
            embeddings = layer(
                embeddings,
                # attention_mask,
                get_attention_weights=get_attention_weights,
                average_attn_weights=average_attn_weights,
                register_hook=register_hook,
            )

        # Get the first token of the sequence which is cls token
        output = self.out_proj(embeddings[:, 0])

        return output


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            device,
            dtype,
        )

        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = CustomMultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            **factory_kwargs,
        )

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        get_attention_weights: bool = False,
        average_attn_weights: bool = False,
        register_hook: bool = False,
    ) -> Tensor:
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ""
        if not src.dim() == 3:
            why_not_sparsity_fast_path = (
                f"input not batched; expected src.dim() of 3 but got {src.dim()}"
            )
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first:
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif not self.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (
            src_key_padding_mask is not None or src_mask is not None
        ):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"  # noqa: E501
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.is_cuda or "cpu" in str(x.device)) for x in tensor_args):
                why_not_sparsity_fast_path = (
                    "some Tensor argument is neither CUDA nor CPU"
                )
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = self.self_attn.merge_masks(
                    src_mask, src_key_padding_mask, src
                )
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    merged_mask,
                    mask_type,
                )

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(
                    x,
                    src_mask,
                    src_key_padding_mask,
                    get_attention_weights=get_attention_weights,
                    average_attn_weights=average_attn_weights,
                    regiester_hook=register_hook,
                )
            )
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        get_attention_weights=False,
        average_attn_weights=False,
        regiester_hook: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=get_attention_weights,
            average_attn_weights=average_attn_weights,
            register_hook=regiester_hook,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
