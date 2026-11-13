"""Convert an SSAST checkpoint to the parameter format used by CustomASTModel.

The paper initialised the sentence-level AST with SSAST-Base-Frame-400
(Gong et al., 2022): https://github.com/YuanGongND/ssast

The SSAST checkpoint uses the original ViT parameter names
(module.v.blocks.N...). This script renames them to the Hugging Face AST
convention, splits the fused qkv projections into query/key/value, drops the
pretraining heads and replaces the positional embeddings with zeros, because
the overlapping frame patches used here give a different number of tokens
(1023 frames + cls + distillation token = 1025).

Usage (from the repository root):
    python scripts/adapt_params.py --ssast-checkpoint SSAST-Base-Frame-400.pth
"""

import argparse
import re
from pathlib import Path

import torch


def split_and_rename_qkv_tensors(updated_state_dict):
    """Separate module.v.blocks.{layer_num}.attn.qkv.weight into 3 separate tensors for query, key and value tensors defined in huggingface."""
    # Create a pattern to match the keys we are interested in
    pattern = re.compile(
        r"module\.v\.blocks\.(\d+)\.attn\.qkv\.(weight|bias)",
    )

    # Iterate over a copy of items, so we can modify the original dict
    for key, params in updated_state_dict.copy().items():
        match = pattern.match(key)
        if match:
            # Extract the layer number and tensor type (weight or bias)
            layer_num, tensor_type = match.groups()

            # Split the tensor
            query_tensor, key_tensor, value_tensor = torch.split(
                params, split_size_or_sections=768, dim=0
            )

            # Construct new key names
            prefix = f"encoder.layer.{layer_num}.attention.attention"
            query_key = f"{prefix}.query.{tensor_type}"
            key_key = f"{prefix}.key.{tensor_type}"
            value_key = f"{prefix}.value.{tensor_type}"

            # Update the dictionary with the split tensors
            updated_state_dict[query_key] = query_tensor
            updated_state_dict[key_key] = key_tensor
            updated_state_dict[value_key] = value_tensor

            # Remove the original vit key
            del updated_state_dict[key]

    return updated_state_dict


def rename_key_for_each_layer(key):
    """Renames keys for each layer of the vit model to huggingface convention."""
    # Define a list of patterns and their corresponding new formats
    patterns = [
        (
            r"module\.v\.blocks\.(\d+)\.norm1\.weight",
            "encoder.layer.{}.layernorm_before.weight",
        ),
        (
            r"module\.v\.blocks\.(\d+)\.norm1\.bias",
            "encoder.layer.{}.layernorm_before.bias",
        ),
        (
            r"module\.v\.blocks\.(\d+)\.norm2\.weight",
            "encoder.layer.{}.layernorm_after.weight",
        ),
        (
            r"module\.v\.blocks\.(\d+)\.norm2\.bias",
            "encoder.layer.{}.layernorm_after.bias",
        ),
        (
            r"module\.v\.blocks\.(\d+)\.attn\.proj\.weight",
            "encoder.layer.{}.attention.output.dense.weight",
        ),
        (
            r"module\.v\.blocks\.(\d+)\.attn\.proj\.bias",
            "encoder.layer.{}.attention.output.dense.bias",
        ),
        (
            r"module\.v\.blocks\.(\d+)\.mlp\.fc1\.weight",
            "encoder.layer.{}.intermediate.dense.weight",
        ),
        (
            r"module\.v\.blocks\.(\d+)\.mlp\.fc1\.bias",
            "encoder.layer.{}.intermediate.dense.bias",
        ),
        (
            r"module\.v\.blocks\.(\d+)\.mlp\.fc2\.weight",
            "encoder.layer.{}.output.dense.weight",
        ),
        (
            r"module\.v\.blocks\.(\d+)\.mlp\.fc2\.bias",
            "encoder.layer.{}.output.dense.bias",
        ),
    ]

    # Try each pattern until a match is found
    for pattern, format_str in patterns:
        match = re.match(pattern, key)
        if match:
            # If pattern matches, reformat the key string
            layer_num = match.group(1)
            return format_str.format(layer_num)

    # If the key doesn't match any pattern, return it unchanged
    return key


def rename_keys_to_huggingface(vit_params):
    """Renames keys of vit model parameters to huggingface convention."""
    updated_state_dict = {
        key: value for key, value in vit_params.items() if "module.v" in key
    }

    # Define a mapping of pytorch vit key names to huggingface param key names
    key_mapping = {
        "module.v.cls_token": "embeddings.cls_token",
        "module.v.dist_token": "embeddings.distillation_token",
        "module.v.pos_embed": "embeddings.position_embeddings",
        "module.v.patch_embed.proj.weight": "embeddings.patch_embeddings.projection.weight",
        "module.v.patch_embed.proj.bias": "embeddings.patch_embeddings.projection.bias",
        "module.v.norm.weight": "layernorm.weight",
        "module.v.norm.bias": "layernorm.bias",
    }

    # Rename the keys for non-layer parameters
    updated_state_dict = {
        key_mapping.get(key, key): value for key, value in updated_state_dict.items()
    }

    # Rename the keys for layer parameters
    updated_state_dict = {
        rename_key_for_each_layer(key): value
        for key, value in updated_state_dict.items()
    }

    updated_state_dict = split_and_rename_qkv_tensors(
        updated_state_dict,
    )

    keys_to_remove = [
        "module.v.head.bias",
        "module.v.head_dist.bias",
        "module.v.head_dist.weight",
        "module.v.head.weight",
    ]

    for key in keys_to_remove:
        if key in updated_state_dict:
            del updated_state_dict[key]

    # Reshape and reinitialise positional embedding tensors, -> [1, 1025, 768]

    embeddings_params = torch.zeros(1, 1025, 768)

    updated_state_dict["embeddings.position_embeddings"] = embeddings_params

    return updated_state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ssast-checkpoint",
        type=Path,
        required=True,
        help="Path to SSAST-Base-Frame-400.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data",
            "pretrained_params",
            "adapted_frame_based_AST_params.pt",
        ),
        help="Where to write the converted parameters",
    )
    args = parser.parse_args()

    frame_based_AST_params = torch.load(
        args.ssast_checkpoint,
        map_location=torch.device("cpu"),
        weights_only=True,
    )

    updated_frame_based_AST_params = rename_keys_to_huggingface(
        frame_based_AST_params,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        updated_frame_based_AST_params,
        args.output,
    )

    print(f"Saved adapted parameters to {args.output}")
