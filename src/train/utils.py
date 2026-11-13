import math
from pathlib import Path

import torch
from load.load import SpeechLevelDataset
from model.SpeechLevelAST import SpeechLevelAST
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

torch.manual_seed(seed=42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed=42)


def get_accuracy(logits, label):
    _, predicted = torch.max(logits, 1)
    correct = (predicted == label).float()
    accuracy = correct.sum() / len(correct)

    print(predicted)
    print(label)

    return accuracy


def calculate_steps(config=None):
    """Calculates the total number of training steps.

    Decide warmup steps for setting up scheduler.
    """
    batch_size = 1
    accumulation_steps = config.accumulation_steps
    epochs = config.epochs
    num_samples = config.train_sample_num

    # Number of mini-batches
    num_batches = num_samples // batch_size  # Use floor division here

    # Number of updates per epoch
    updates_per_epoch = math.ceil(
        num_batches / accumulation_steps
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

    # Speech-level Transformer Encoder parameters
    named_parameters_transformer_encoder = list(
        model.speech_level_block.named_parameters()
    )
    params = [p for _, p in named_parameters_transformer_encoder]

    opt_parameters.append(
        {"params": params, "lr": lr_ast, "weight_decay": weight_decay}
    )

    # Sentence-level parameters
    named_parameters_ast = list(
        model.sentence_level_block.module.encoder.named_parameters()
    )

    # Layernorm parameters
    params = [p for n, p in named_parameters_ast if "layernorm." in n]
    opt_parameters.append({"params": params, "lr": lr_ast})

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
            lr_ast *= 0.9
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
        model.sentence_level_block.module.encoder.named_parameters()
    )

    params = [p for n, p in named_parameters_ast if "embeddings" in n]
    for param in params:
        param.requires_grad = False

    # Encoder layer parameters
    for layer in range(0, 9):
        params = [p for n, p in named_parameters_ast if f"encoder.layer.{layer}." in n]
        for param in params:
            param.requires_grad = False


def build_loader():
    # Read the data
    dataset = SpeechLevelDataset(
        annotations_file_path=Path("annotations", "train_speech_level.csv"),
        speech_level_data_path=Path("data", "speech_level_train"),
        waveform_normalize=True,
    )

    val_dataset = SpeechLevelDataset(
        annotations_file_path=Path("annotations", "test_speech_level.csv"),
        speech_level_data_path=Path("data", "speech_level_test"),
        waveform_normalize=True,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=True,
        # sampler=train_sampler,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=None,
    )

    return train_loader, val_loader


def build_model(
    config=None,
    device=None,
):
    """Builds the model from the config file."""
    reinit_encoders_n_layers = config.reinit_encoders_n_layers
    num_layers_pl_encoder = config.num_layers_pl_encoder

    model = SpeechLevelAST(
        reinit_encoders_n_layers=reinit_encoders_n_layers,
        num_layers_pl_encoder=num_layers_pl_encoder,
    ).to(device)

    requires_no_grad(model=model)

    return model


def train_one_epoch(
    config=None,
    model=None,
    train_loader=None,
    optimizer=None,
    scheduler=None,
    device=None,
):
    # Set model to training mode
    model.train()

    logits_list = []

    labels_list = []

    total_loss = 0.0

    # Iterate over batches of data
    for i, batch in enumerate(train_loader, start=1):
        # Move batch tensors to the same device as the model
        audio_inputs = batch["audio_inputs"].to(device)
        audio_attn_masks = batch["audio_attn_masks"].to(device)
        target = batch["label"].to(device)

        # Forward pass
        outputs = model(
            audio_inputs=audio_inputs,
            audio_attn_masks=audio_attn_masks,
        )

        # Compute loss for the batch
        loss = torch.nn.functional.cross_entropy(outputs, target)

        logits_list.append(outputs.detach().cpu())

        labels_list.append(target.detach().cpu())

        batch_loss = loss.detach()

        # Sum batch loss up for later calculation of average epoch loss
        total_loss += batch_loss

        # Normalize loss by the number of accumulation steps before backward pass
        loss = loss / config.accumulation_steps

        loss.backward()

        # Accumulate gradients to update weights every accumulation_steps
        if i % config.accumulation_steps == 0 or i == len(train_loader):
            logits = torch.cat(logits_list, dim=0)
            labels = torch.cat(labels_list, dim=0)

            accuracy = get_accuracy(logits, labels).cpu()

            print(f"train batch loss: {batch_loss:.4f}, accuracy: {accuracy:.4f}")

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # Return average loss
    return total_loss / len(train_loader)


def validate(
    model=None,
    validation_loader=None,
    device=None,
):
    # Set model to evaluation mode
    model.eval()

    logits_list = []
    labels_list = []

    total_loss = 0.0

    # Disable gradient computations
    with torch.no_grad():
        # Iterate over batches of data
        for batch in validation_loader:
            # Move batch tensors to the same device as the model
            audio_inputs = batch["audio_inputs"].to(device)
            audio_attn_masks = batch["audio_attn_masks"].to(device)
            target = batch["label"].to(device)

            # Forward pass
            outputs = model(
                audio_inputs=audio_inputs,
                audio_attn_masks=audio_attn_masks,
            )

            # Compute loss
            loss = torch.nn.functional.cross_entropy(outputs, target)

            # Add up the validation loss
            total_loss += loss.item()

            logits_list.append(outputs.detach().cpu())
            labels_list.append(target.detach().cpu())

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    accuracy = get_accuracy(logits, labels).cpu()

    print(f"validation accuracy: {accuracy:.4f}")

    # Return average loss
    return total_loss / len(validation_loader)


def train(
    config,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        config,
        device=device,
    )

    total_steps, warmup_steps = calculate_steps(config=config)

    optimizer, scheduler = build_optimizer_scheduler(
        model=model,
        config=config,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    train_loader, val_loader = build_loader()

    for epoch in range(config.epochs):
        avg_loss = train_one_epoch(
            config=config,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        # Call the validation function
        avg_val_loss = validate(
            model=model,
            validation_loader=val_loader,
            device=device,
        )

        print(
            f"epoch {epoch + 1}: train loss {avg_loss:.4f}, "
            f"validation loss {avg_val_loss:.4f}"
        )

        if epoch + 1 > 5:
            torch.save(model.state_dict(), f"model_weights{epoch + 1}.pt")
