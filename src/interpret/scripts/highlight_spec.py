
import matplotlib.pyplot as plt

# import numpy as np
import torch
from torch import tensor


def plot_spec_with_pads(
    spec: tensor,
    title: str = None,
    ylabel="freq_bin",
):
    """Visualize log_spectrogram in decibel scale with paddings."""
    transposed_spec = spec.squeeze().transpose(0, 1)
    _, axs = plt.subplots(1, 1)
    axs.set_title(title or "Log_ Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    axs.imshow(transposed_spec.numpy(), origin="lower", aspect="auto")
    # fig.colorbar(im, ax=axs)
    plt.show(block=False)


def plot_spec_without_pads(
    # axs,  # Note: passing axis
    spec: tensor,
    title: str = "Original spectrogram",
    ylabel="Frequency bin",
):
    """Visualize log_spectrogram in decibel scale with paddings removed. Paddings were added by the AST FeatureExtractor, sometimes the audio is too short so its better to remove the paddings before plotting."""
    transposed_spec = spec.transpose(0, 1)

    # Remove paddings
    spec_without_pads = transposed_spec[:, transposed_spec.sum(axis=0) != 0]

    _, axs = plt.subplots(1, 1)
    axs.set_title(title or "Log_ Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(spec_without_pads.numpy(), origin="lower", aspect="auto")
    plt.colorbar(im, ax=axs, label="Intensity")
    # plt.show(block=False)


def plot_relevancy_map_with_pads(
    spec: torch.Tensor,
    relevancy_map: torch.Tensor,
    title: str = None,
    ylabel="freq_bin",
):
    """Visualize log_spectrogram in decibel scale with relevancy map as alpha channel to reflect attention from the model for classification."""
    # Transpose the spectrogram and convert to numpy
    spec_with_pads = spec.squeeze().transpose(0, 1).numpy()

    # Normalize the relevancy map to be between 0 and 1 and convert to numpy
    relevancy_map_np = relevancy_map.squeeze().numpy()
    relevancy_map_np = (relevancy_map_np - relevancy_map_np.min()) / (
        relevancy_map_np.max() - relevancy_map_np.min()
    )

    # Create an RGBA version of the spectrogram with the relevancy map as the alpha channel

    # spectrogram[..., 3] = relevancy_map_np

    # Plot
    _, axs = plt.subplots(1, 1)
    axs.set_title(title or "Log_ Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    axs.imshow(
        spec_with_pads,
        origin="lower",
        aspect="auto",
        alpha=relevancy_map_np,
    )
    # fig.colorbar(im, ax=axs)
    plt.show(block=False)


def plot_relevancy_map_without_pads(
    # axs,  # Note: passing axis
    spec: torch.Tensor,
    relevancy_map: torch.Tensor,
    title: str = "Depression classification relevant frames",
    ylabel="Frequency bin",
):
    # Transpose the spectrogram and convert to numpy
    spec = spec.transpose(0, 1)

    spec_without_pads = spec[:, spec.sum(axis=0) != 0].numpy()

    relevancy_map_np = relevancy_map.squeeze().numpy()

    relevancy_map_without_pads = relevancy_map_np[:, spec.sum(axis=0) != 0]

    # Normalize the relevancy map to be between 0 and 1 and convert to numpy after removing the paddings
    normalised_relevancy_map = (
        relevancy_map_without_pads - relevancy_map_without_pads.min()
    ) / (relevancy_map_without_pads.max() - relevancy_map_without_pads.min())

    _, axs = plt.subplots(1, 1)
    axs.set_title(title or "Log_ Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(
        spec_without_pads,
        origin="lower",
        aspect="auto",
        alpha=normalised_relevancy_map,
    )
    plt.colorbar(im, ax=axs, label="Intensity")
    # plt.show(block=False)


def map_relevancy_to_spec_size(relevancy: torch.tensor):
    # Before flatterning, the sequence of spectrogram patches has the shape (1, 768, 12, 101), and the spectrogram has the shape (1, 128, 1024)
    batch_size = 1
    num_patches_freq = 1
    num_patches_time = 1023
    frequency_shape = 128
    time_shape = 2
    frequency_stride = 128
    time_stride = 1
    spec_height = 128
    spec_width = 1024

    # Remove the cls and distill tokens related relevancy and unsequeeze to be ready for the repeating operation
    relevancy_map = relevancy.unsqueeze(1).unsqueeze(1)

    # Repeat the relevancy scores to match the size of frame-based patches and concatenate the relevancy scores along the time dimension (result size torch.Size([1, 128, 2046]))

    repeated_relevancy = relevancy_map.repeat(1, 128, 2)

    upscaled_relevancy = torch.cat(
        repeated_relevancy.unbind(dim=0),
        dim=1,
    ).unsqueeze(0)

    # Initialize the reconstructed relevancy map to match the spectrogram size
    reconstructed_relevancy = torch.zeros(
        batch_size,
        spec_height,
        spec_width,
    )

    # Initialize a overlap map to keep track of the overlapping patches
    overlap_map = torch.zeros(
        batch_size,
        spec_height,
        spec_width,
    )

    # Because patches were sampled with overlapped during training, and so far the upscale relevancy maps ignore the overlapping, we sample each patch of relevancy scores from the upscaled relevancy map to construct a relevancy map that has the same size as the spectrogram, and the relevancy scores of the overlapping areas are averaged.
    for freq_step in range(num_patches_freq):
        for time_step in range(num_patches_time):
            start_freq_step_upscaled = freq_step * frequency_shape
            start_time_step_upscaled = time_step * time_shape
            end_freq_step_upscaled = start_freq_step_upscaled + frequency_shape
            end_time_step_upscaled = start_time_step_upscaled + time_shape

            start_freq_step_reconstructed = freq_step * frequency_stride
            start_time_step_reconstructed = time_step * time_stride
            end_freq_step_reconstructed = (
                start_freq_step_reconstructed + frequency_shape
            )
            end_time_step_reconstructed = start_time_step_reconstructed + time_shape

            # Error check to ensure patch dimensions are consistent
            assert end_freq_step_upscaled <= upscaled_relevancy.size(
                1
            ) and end_time_step_upscaled <= upscaled_relevancy.size(
                2
            ), "Patch dimensions are inconsistent with the upscaled image size."

            assert (
                end_freq_step_reconstructed <= spec_height
                and end_time_step_reconstructed <= spec_width
            ), "Patch dimensions are inconsistent with the original image size."

            # Extract out the patch from the upsampled relevancy map
            patch = upscaled_relevancy[
                0,
                start_freq_step_upscaled:end_freq_step_upscaled,
                start_time_step_upscaled:end_time_step_upscaled,
            ]

            reconstructed_relevancy[
                0,
                start_freq_step_reconstructed:end_freq_step_reconstructed,
                start_time_step_reconstructed:end_time_step_reconstructed,
            ] += patch

            overlap_map[
                0,
                start_freq_step_reconstructed:end_freq_step_reconstructed,
                start_time_step_reconstructed:end_time_step_reconstructed,
            ] += 1

    reconstructed_relevancy /= overlap_map

    reconstructed_relevancy[torch.isnan(reconstructed_relevancy)] = 0

    return reconstructed_relevancy
