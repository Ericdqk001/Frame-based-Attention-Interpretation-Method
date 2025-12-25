import matplotlib.pyplot as plt
import numpy as np
import torch
from interpret.scripts.highlight_spec import (
    map_relevancy_to_spec_size,
    plot_relevancy_map_with_pads,
    plot_spec_with_pads,
)

# from load.load_local import SpeechLevelDataset


def interpret_speech_level(
    generator,
    num_sentences_indexes,
):
    """Returns the indexes of the sentences with the highest relevancies.

    Returns:
        _type_: _description_
    """
    speech_level_interpretation = generator.generate_speech_level_interpretation()

    # Sort the tensor in descending order and get the indexes
    _, indexes = torch.sort(speech_level_interpretation, descending=True)

    # Get the indexes of the first sentences
    top_indexes = indexes[:num_sentences_indexes]

    return top_indexes


def interpret_sentence_level(
    generator,
    sentence_index,
    audio_inputs,
):
    """Plots the original spectrogram and the reconstructed relevancy map."""
    sentence_level_interpretation = generator.generate_sentence_level_explanations(
        sentence_index=sentence_index
    )

    reconstructed_relevancy = map_relevancy_to_spec_size(sentence_level_interpretation)

    # _, axs = plt.subplots(1, 2, figsize=(10, 5))

    plot_spec_with_pads(
        # axs[0],
        audio_inputs[sentence_index],
    )

    plot_relevancy_map_with_pads(
        # axs[1],
        spec=audio_inputs[sentence_index],
        relevancy_map=reconstructed_relevancy,
    )

    # plt.tight_layout()  # adjust spacing between subplots
    # plt.show(block=False)


def process_spec(spec: torch.Tensor):
    # Transpose the spectrogram
    transposed_spec = spec.transpose(0, 1)

    # Remove paddings
    spec_without_pads = transposed_spec[:, transposed_spec.sum(axis=0) != 0]

    return spec_without_pads


def process_relevancy_map(
    generator,
    sentence_index,
    spec: torch.Tensor,
):
    spec = spec.transpose(0, 1)

    sentence_level_interpretation = generator.generate_sentence_level_explanations(
        sentence_index=sentence_index
    )

    reconstructed_relevancy = map_relevancy_to_spec_size(sentence_level_interpretation)

    relevancy_map = reconstructed_relevancy.squeeze().numpy()

    # Remove paddings from the spectrogram and relevancy map
    relevancy_map_without_pads = relevancy_map[:, spec.sum(axis=0) != 0]

    # Normalize the relevancy map to be between 0 and 1
    normalised_relevancy_map = (
        relevancy_map_without_pads - relevancy_map_without_pads.min()
    ) / (relevancy_map_without_pads.max() - relevancy_map_without_pads.min())

    return normalised_relevancy_map


def concatenate_and_plot(specs, relevancy_maps, sentence_indexes, threshold=0.3):
    specs = [torch.tensor(spec) for spec in specs]
    relevancy_maps = [torch.tensor(relevancy_map) for relevancy_map in relevancy_maps]

    # Initialize lists for the relevant specs and relevancy maps
    relevant_specs = []
    relevant_relevancy_maps = []

    # Loop over the range of specs and add only the relevant ones or zeros
    for i in range(len(specs)):
        if i in sentence_indexes:
            relevant_specs.append(specs[i])
            relevant_relevancy_maps.append(relevancy_maps[i])
        else:
            zero_spec = torch.zeros_like(specs[i])
            zero_relevancy_map = torch.zeros_like(relevancy_maps[i])
            relevant_specs.append(zero_spec)
            relevant_relevancy_maps.append(zero_relevancy_map)

    concatenated_spec = torch.cat(specs, axis=1)
    concatenated_relevancy_map = torch.cat(relevancy_maps, axis=1)
    concatenated_relevant_spec = torch.cat(relevant_specs, axis=1)
    concatenated_relevant_relevancy_map = torch.cat(relevant_relevancy_maps, axis=1)

    # Apply threshold and create masked spectrogram
    masked_spec = np.where(concatenated_relevancy_map > threshold, concatenated_spec, 1)
    # masked_relevant_spec = np.where(
    #     concatenated_relevant_relevancy_map > threshold, concatenated_relevant_spec, 1
    # )

    alpha_mask = torch.where(
        concatenated_relevant_relevancy_map > 0.3,
        torch.ones_like(concatenated_relevant_relevancy_map),
        torch.zeros_like(concatenated_relevant_relevancy_map),
    )

    # Plot the concatenated spectrogram with thresholded relevancy
    plt.figure(figsize=(len(specs) * 5, 5))
    plt.imshow(
        masked_spec,
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )

    plt.title("Concatenated Original Spectrograms with Thresholded Relevancy")
    plt.ylabel("Frequency bin")
    plt.xlabel("Frame")
    plt.colorbar(label="Intensity")
    plt.show()

    # Plot the concatenated relevant spec with thresholded relevancy
    plt.figure(figsize=(len(relevant_specs) * 5, 5))
    plt.imshow(
        concatenated_relevant_spec,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        alpha=alpha_mask,
    )

    plt.title("Concatenated Relevant Spectrograms with Thresholded Relevancy")
    plt.ylabel("Frequency bin")
    plt.xlabel("Frame")
    # plt.colorbar(label="Intensity")
    plt.show()
