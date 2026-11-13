import math
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from load.load import SegmentLevelDataset
from model.SentenceLevelAST import SentenceLevelAST

torch.manual_seed(seed=42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed=42)


def calculate_steps(config=None):
    """Calculates the total number of training steps and decide warmup steps.

    For setting up scheduler.
    """
    batch_size = config.batch_size
    epochs = config.epochs
    num_samples = config.train_sample_num

    # Number of updates per epoch
    updates_per_epoch = math.ceil(
        num_samples / batch_size
    )  # Use regular division and round up

    # Total number of training steps
    total_steps = updates_per_epoch * epochs

    warmup_steps = total_steps * 0.1

    return total_steps, warmup_steps


def build_optimizer_scheduler(
    model=None,
    config=None,
    total_steps=None,
    warmup_steps=None,
):
    """Builds the optimizer for the model using the training config."""
    lr_ast = config.lr_ast

    weight_decay = config.weight_decay

    opt_parameters = []

    # Sentence-level parameters
    named_parameters_ast = list(model.named_parameters())

    # Output projection parameters
    params = [p for n, p in named_parameters_ast if "out_proj" in n]

    opt_parameters.append(
        {"params": params, "lr": lr_ast, "weight_decay": weight_decay}
    )

    # Layernorm parameters
    params = [p for n, p in named_parameters_ast if "layernorm." in n]
    opt_parameters.append(
        {"params": params, "lr": lr_ast, "weight_decay": weight_decay}
    )

    # lr_ast *= 0.9
    # Encoder layer parameters
    for layer in range(11, -1, -1):
        params = [p for n, p in named_parameters_ast if f"layer.{layer}." in n]
        opt_parameters.append(
            {
                "params": params,
                "lr": lr_ast,
                "weight_decay": weight_decay,
            }
        )
        if layer < 9:
            lr_ast = 0
            weight_decay = 0

    # Embedding parameters
    params = [p for n, p in named_parameters_ast if "embeddings" in n]
    opt_parameters.append(
        {
            "params": params,
            "lr": lr_ast,
            "weight_decay": 0,
        }
    )

    # Initialize optimizer and scheduler
    optimizer = Adam(opt_parameters)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    return optimizer, scheduler


def requires_no_grad(model=None):
    named_parameters_ast = list(
        model.named_parameters(),
    )

    params = [p for n, p in named_parameters_ast if "embeddings" in n]

    for param in params:
        param.requires_grad = False

    # Encoder layer parameters
    for layer in range(0, 9):
        params = [p for n, p in named_parameters_ast if f"encoder.layer.{layer}." in n]
        for param in params:
            param.requires_grad = False


def build_loader(config=None):
    dataset = SegmentLevelDataset(
        annotations_file_path=Path("annotations", "train_segment_level_42.csv"),
        segment_level_data_path=Path("data", "segment_level_train"),
        waveform_normalize=True,
        mean=-2.9899,
        std=4.3509,
    )

    val_dataset = SegmentLevelDataset(
        annotations_file_path=Path("annotations", "test_segment_level_42.csv"),
        segment_level_data_path=Path("data", "segment_level_test"),
        waveform_normalize=True,
        mean=-2.9899,
        std=4.3509,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    return train_loader, val_loader


def build_model(
    config=None,
    device=None,
):
    """Builds the model from the config file."""
    reinit_encoders_n_layers = config.reinit_encoders_n_layers

    model = SentenceLevelAST(
        reinit_encoders_n_layers=reinit_encoders_n_layers,
    ).to(device)

    requires_no_grad(model)

    return model


def train_one_epoch(
    model=None,
    train_loader=None,
    optimizer=None,
    scheduler=None,
    device=None,
):
    # Set model to training mode
    model.train()

    total_loss = 0.0

    # Iterate over batches of data
    for i, batch in enumerate(train_loader, start=1):
        # Move batch tensors to the same device as the model
        audio_inputs, attention_mask, target = (
            batch["audio_inputs"].to(device),
            batch["audio_attn_masks"].to(device),
            batch["label"].to(device),
        )

        # Forward pass
        outputs = model(
            audio_inputs,
            attention_mask,
        )

        # Compute loss
        loss = torch.nn.functional.cross_entropy(outputs, target)

        # Record loss
        batch_loss = loss.detach()

        total_loss += batch_loss

        # Backward pass
        loss.backward()

        print(f"train batch loss: {batch_loss:.4f}")

        # Try to free up memory
        torch.cuda.empty_cache()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Return average loss
    return total_loss / len(train_loader)


def validate(
    model=None,
    val_loader=None,
    device=None,
):
    # Set model to evaluation mode
    model.eval()

    total_loss = 0.0

    # Disable gradient computations
    with torch.no_grad():
        # Iterate over batches of data
        for batch in val_loader:
            # Move batch tensors to the same device as the model
            audio_inputs, attention_mask, target = (
                batch["audio_inputs"].to(device),
                batch["audio_attn_masks"].to(device),
                batch["label"].to(device),
            )

            # Forward pass
            outputs = model(
                audio_inputs,
                attention_mask,
            )

            # Compute loss
            loss = torch.nn.functional.cross_entropy(outputs, target)

            # Add up the validation loss
            total_loss += loss.item()

            torch.cuda.empty_cache()

    # Return average loss
    return total_loss / len(val_loader)


def train(
    config,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        config,
        device=device,
    )

    total_steps, warmup_steps = calculate_steps(
        config=config,
    )

    optimizer, scheduler = build_optimizer_scheduler(
        model=model,
        config=config,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    train_loader, val_loader = build_loader(
        config=config,
    )

    for epoch in range(config.epochs):
        avg_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        # Call the validation function
        avg_val_loss = validate(
            model=model,
            val_loader=val_loader,
            device=device,
        )

        print(
            f"epoch {epoch + 1}: train loss {avg_loss:.4f}, "
            f"validation loss {avg_val_loss:.4f}"
        )

        if epoch + 1 > 5:
            torch.save(model.state_dict(), f"seg_model_weights{epoch + 1}.pt")
