import math

import psutil
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

import wandb
from load.load import SegmentLevelDataset, SegmentLevelDatasetTest
from model.SentenceLevelAST import SentenceLevelAST

torch.manual_seed(seed=42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed=42)


def check_cpu_memory():
    # Get CPU utilization as a percentage
    cpu_percent = psutil.cpu_percent(interval=1)
    print("CPU percent used:", cpu_percent, "%")

    # Get information about system memory usage
    vm_info = psutil.virtual_memory()
    print("Total virtual memory:", vm_info.total / (1024**3), "GB")
    print("Available virtual memory:", vm_info.available / (1024**3), "GB")
    print("Used virtual memory:", vm_info.used / (1024**3), "GB")
    print("Memory percent used:", vm_info.percent, "%")

    print("RAM memory % used:", psutil.virtual_memory()[2])
    # Getting usage of virtual_memory in GB ( 4th field)
    print("RAM Used (GB):", psutil.virtual_memory()[3] / 1000000000)

    # Return a dictionary of CPU and memory information
    return {
        "RAM memory % used": psutil.virtual_memory()[2],
        "RAM Used (GB)": psutil.virtual_memory()[3] / 1000000000,
    }


def check_mps_memory():
    process = psutil.Process()
    mem_info = process.memory_info()
    print("Memory used:", mem_info.rss / 1024 / 1024 / 1024, "GB")
    print("Virtual memory used:", mem_info.vms / 1024 / 1024 / 1024, "GB")

    vm_info = psutil.virtual_memory()
    print("Total virtual memory:", vm_info.total / 1024 / 1024 / 1024, "GB")
    print("Available virtual memory:", vm_info.available / 1024 / 1024 / 1024, "GB")
    print("Used virtual memory:", vm_info.used / 1024 / 1024 / 1024, "GB")
    print("Memory percent used:", vm_info.percent, "%")
    print("---------")

    return {
        "memory used": mem_info.rss / 1024 / 1024 / 1024,
        "virtual memory used": mem_info.vms / 1024 / 1024 / 1024,
        "total virtual memory": vm_info.total / 1024 / 1024 / 1024,
        "available virtual memory": vm_info.available / 1024 / 1024 / 1024,
        "used virtual memory": vm_info.used / 1024 / 1024 / 1024,
        "memory percent used": vm_info.percent,
    }


def check_memory():
    print(
        "torch.cuda.memory_allocated: %fGB"
        % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
    )
    print(
        "torch.cuda.memory_reserved: %fGB"
        % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024)
    )
    print(
        "torch.cuda.max_memory_reserved: %fGB"
        % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024)
    )
    print("---------")

    return {
        "memory allocated": torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024,
        "memory reserved": torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024,
        "max memory reserved": torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024,
    }


def get_accuracy(logits, label):
    _, predicted = torch.max(logits, 1)
    correct = (predicted == label).float()
    accuracy = correct.sum() / len(correct)

    return accuracy


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
    """Builds the optimizer for the model using wandb sweep config."""
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
        waveform_normalize=True,
        mean=-2.9899,
        std=4.3509,
    )

    val_dataset = SegmentLevelDatasetTest(
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
        wandb.log(check_memory())

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

        wandb.log({"train batch loss": batch_loss})

        total_loss += batch_loss

        # Backward pass
        loss.backward()

        accuracy = get_accuracy(outputs, target).cpu()

        wandb.log({"train batch accuracy": accuracy})

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

    logits_list = []
    labels_list = []

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

            logits_list.append(outputs.detach().cpu())
            labels_list.append(target.detach().cpu())

            torch.cuda.empty_cache()

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    accuracy = get_accuracy(logits, labels).cpu()

    wandb.log({"validation accuracy": accuracy})

    # Return average loss
    return total_loss / len(val_loader)


def train(
    wandb,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        wandb.config,
        device=device,
    )

    # Create a wandb artifact for the best model
    model_artifact = wandb.Artifact(
        wandb.run.name, type="model"
    )  # Use run name as the artifact name

    total_steps, warmup_steps = calculate_steps(
        config=wandb.config,
    )

    optimizer, scheduler = build_optimizer_scheduler(
        model=model,
        config=wandb.config,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    train_loader, val_loader = build_loader(
        config=wandb.config,
    )

    for epoch in range(wandb.config.epochs):
        avg_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        wandb.log({"train epoch loss": avg_loss, "epoch": epoch})

        # Call the validation function
        avg_val_loss = validate(
            model=model,
            val_loader=val_loader,
            device=device,
        )

        wandb.log({"validation loss": avg_val_loss, "epoch": epoch})

        if epoch + 1 > 5:
            torch.save(model.state_dict(), f"seg_model_weights{epoch + 1}.pt")

            model_artifact.add_file(f"seg_model_weights{epoch + 1}.pt")

    wandb.run.log_artifact(model_artifact)
