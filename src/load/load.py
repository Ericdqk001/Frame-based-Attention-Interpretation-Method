# %%
import json
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import ASTFeatureExtractor


class SpeechLevelDataset(Dataset):
    def __init__(
        self,
        annotations_file_path=Path("annotations", "test_speech_level.csv"),
        speech_level_data_path=Path("data", "speech_level_test"),
        sampling_rate=16000,
        waveform_normalize=True,
        mean=-4.1389,
        std=5.3054,
        id_sentences_frames_path=None,
        num_sentences_to_remove=None,
        true_positive_ids=None,
        true_negative_ids=None,
    ):
        self.annotations = pd.read_csv(annotations_file_path)

        self.speech_level_data_path = Path(speech_level_data_path)

        self.sampling_rate = sampling_rate

        # Maximum number of recordings (sentences) used per participant
        self.batch_size = 42

        self.waveform_normalize = waveform_normalize

        self.mean = mean

        self.std = std

        self.waveform_zero = None

        self.extractor = ASTFeatureExtractor(
            do_normalize=False,
        )

        if true_positive_ids and true_negative_ids:
            self.ids_to_permute = true_positive_ids + true_negative_ids

        if id_sentences_frames_path:
            with open(id_sentences_frames_path) as f:
                self.id_sentences_frames = json.load(f)
                self.num_sentences_to_remove = num_sentences_to_remove
        else:
            self.id_sentences_frames = None
            self.num_sentences_to_remove = None

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        id = str(
            self.annotations.loc[index, "id"],
        )
        speech_level_data_folder = Path(
            self.speech_level_data_path,
            id,
        )

        # Sort the waveforms by their start time because positions matter for the
        # Speech-level model
        speech_level_paths = sorted(
            speech_level_data_folder.glob("*.wav"),
            key=lambda p: float(p.stem.split("-")[-2]),
        )

        waveforms = []

        original_sampling_rate = torchaudio.load(speech_level_paths[0])[1]

        for j, waveform_path in enumerate(speech_level_paths):
            if j < self.batch_size:
                waveform = torchaudio.load(waveform_path)[0]
                waveforms.append(waveform)

        # Permutation test
        if self.id_sentences_frames:
            if id in self.ids_to_permute:
                sentences_frames = self.id_sentences_frames[id]
                for i, sentence_index in enumerate(sentences_frames):
                    if i < self.num_sentences_to_remove:
                        for frame_index in sentences_frames[sentence_index]:
                            waveforms[int(sentence_index)] = self.silence_waveform(
                                waveforms[int(sentence_index)],
                                frame_second=round(
                                    self.convert_frame_to_seconds(
                                        frame_index,
                                    ),
                                    3,
                                ),
                            )

        label = self.annotations.loc[index, "label"]

        label = torch.tensor(label).unsqueeze(0)

        # resample the waveforms to match the required sampling rate (16000)
        # by AST model
        if original_sampling_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sampling_rate,
                new_freq=self.sampling_rate,
            )
            waveforms = [resampler(waveform) for waveform in waveforms]

        # Use the AST FeatureExtractor to prepare the inputs for the AST model,
        # which are log-mel spectrograms

        waveforms_np = [waveform.squeeze().numpy() for waveform in waveforms]

        audio_inputs_list = [
            self.extractor(
                waveform,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
            )["input_values"]
            for waveform in waveforms_np
        ]

        # Prepare attention masks based on the paddings of the spectrograms

        audio_attn_mask_list = []

        for audio_input in audio_inputs_list:
            attention_mask = self.generate_attention_mask(audio_input)
            audio_attn_mask_list.append(attention_mask)

        # Normalise after generating attn_masks to avoid normalising the paddings

        if self.waveform_normalize:
            audio_inputs_list = [
                ((audio_input - self.mean) / self.std) * 0.5
                for audio_input in audio_inputs_list
            ]

        audio_inputs = torch.cat(audio_inputs_list, dim=0)

        audio_attn_masks = torch.cat(audio_attn_mask_list, dim=0)

        return {
            "audio_inputs": audio_inputs[: self.batch_size],
            "audio_attn_masks": audio_attn_masks[: self.batch_size],
            "label": label,
            "id": id,
        }

    def load_audio(
        self,
        waveform_path: str,
    ) -> torch.tensor:
        with self.file_system.open(waveform_path) as f:
            waveform, sample_rate = torchaudio.load(f)

        return waveform, sample_rate

    def silence_waveform(
        self,
        waveform: torchaudio,
        frame_second,
        sample_rate: int = 44100,
        hamming_window_time: int = 0.025,
    ):
        """Remove the relvant frames from the waveform.

        Hamming window was 25ms, so 0.025s of waveform is removed

        """
        if self.waveform_zero is None:
            self.waveform_zero = torch.zeros_like(waveform)

        start_silence = int(sample_rate * frame_second)
        end_silence = start_silence + int(sample_rate * hamming_window_time)

        waveform[:, start_silence:end_silence] = 0

        self.waveform_zero[:, start_silence:end_silence] = 1

        return waveform

    def convert_frame_to_seconds(
        self,
        frame_index: int,
        frame_time_length: int = 0.01,
    ) -> float:
        """Locate the time point in waveform given the number of the frame."""
        return frame_index * frame_time_length

    def generate_attention_mask(self, input_values: torch.Tensor) -> torch.Tensor:
        frequency_shape = 128
        frequency_stride = 10
        time_shape = 2
        time_stride = 1

        # Transpose the input values to (batch_size, frequency, time_length),
        # same procedure done by the embedding layer
        transposed_input_values = input_values.transpose(1, 2)

        # Unfold the input values to (batch_size, num_patches_frequency,
        # num_patches_time, patch_size, patch_size)
        unfolded_patches = transposed_input_values.unfold(
            1, frequency_shape, frequency_stride
        ).unfold(2, time_shape, time_stride)

        # Flatten the last two dimensions to (batch_size, num_patches_frequency,
        # num_patches_time, patch_size * patch_size)
        patches_flat = unfolded_patches.flatten(-2, -1)

        # Permute the dimensions to (batch_size, patch_size * patch_size,
        # num_patches_frequency, num_patches_time)
        patches_perm = patches_flat.permute(0, 3, 1, 2)

        # Flatten the last two dimensions to get total number of patches and transpose
        # to (batch_size, num_patches, patch_size * patch_size)
        patches_final = patches_perm.flatten(-2, -1).transpose(1, 2)

        # Sum the values of each patch to get a tensor of shape
        # (batch_size, num_patches)
        patches_final_sum = patches_final.sum(-1)

        # Create the attention mask with 0s for padding patches and 1s for non-padding
        # patches, similar to the attention mask created by BertTokenizer
        attention_mask = (patches_final_sum != 0).long()

        ones = torch.ones(1, 2)

        attention_mask = torch.cat(
            (ones, attention_mask),
            dim=1,
        )

        return attention_mask


class SegmentLevelDataset(Dataset):
    def __init__(
        self,
        annotations_file_path=Path("annotations", "test_segment_level_42.csv"),
        segment_level_data_path=Path("data", "segment_level_test"),
        sampling_rate=16000,
        waveform_normalize=False,
        mean=-4.1389,
        std=5.3054,
    ):
        self.sampling_rate = sampling_rate

        self.segment_level_annotations = pd.read_csv(annotations_file_path)

        self.segment_level_data_path = Path(segment_level_data_path)

        self.waveform_normalize = waveform_normalize

        self.mean = mean

        self.std = std

        self.extractor = ASTFeatureExtractor(
            do_normalize=False,
        )

    def __len__(self):
        return len(self.segment_level_annotations)

    def __getitem__(self, index):
        waveform_path = Path(
            self.segment_level_data_path,
            self.segment_level_annotations.loc[index, "file_stem"] + ".wav",
        )

        segment_id = self.segment_level_annotations.loc[index, "id"]

        label = self.segment_level_annotations.loc[index, "label"]

        waveform, original_sampling_rate = torchaudio.load(
            waveform_path,
        )

        # resample the waveforms to match the required sampling rate (16000) by AST model
        if original_sampling_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sampling_rate,
                new_freq=self.sampling_rate,
            )
            waveform = resampler(waveform)

        # Uses the AST FeatureExtractor to prepare the inputs for the AST model,
        # which are log-mel spectrograms, the waveform size is [1, ...] so it is squeezed
        audio_input = self.extractor(
            waveform.squeeze().numpy(),
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )

        # Normalise after generating attn_masks to avoid normalising the paddings
        attention_mask = self.generate_attention_mask(
            audio_input["input_values"]
        ).squeeze(0)

        if self.waveform_normalize:
            audio_input["input_values"] = (
                audio_input["input_values"] - self.mean
            ) / self.std

        # Remove the batch dimension because dataloader will add it.
        audio_input["input_values"] = audio_input["input_values"].squeeze(0)

        return {
            "audio_inputs": audio_input["input_values"],
            "audio_attn_masks": attention_mask,
            "label": label,
            "id": segment_id,
        }

    def generate_attention_mask(self, input_values: torch.Tensor) -> torch.Tensor:
        frequency_shape = 128
        frequency_stride = 10
        time_shape = 2
        time_stride = 1

        # Transpose the input values to (batch_size, frequency, time_length),
        # same procedure done by the embedding layer
        transposed_input_values = input_values.transpose(1, 2)

        # Unfold the input values to (batch_size, num_patches_frequency,
        # num_patches_time, patch_size, patch_size)
        unfolded_patches = transposed_input_values.unfold(
            1, frequency_shape, frequency_stride
        ).unfold(2, time_shape, time_stride)

        # Flatten the last two dimensions to (batch_size, num_patches_frequency,
        # num_patches_time, patch_size * patch_size)
        patches_flat = unfolded_patches.flatten(-2, -1)

        # Permute the dimensions to (batch_size, patch_size * patch_size,
        # num_patches_frequency, num_patches_time)
        patches_perm = patches_flat.permute(0, 3, 1, 2)

        # Flatten the last two dimensions to get total number of patches and transpose
        # to (batch_size, num_patches, patch_size * patch_size)
        patches_final = patches_perm.flatten(-2, -1).transpose(1, 2)

        # Sum the values of each patch to get a tensor of shape
        # (batch_size, num_patches)
        patches_final_sum = patches_final.sum(-1)

        # Create the attention mask with 0s for padding patches and 1s for non-padding
        # patches, similar to the attention mask created by BertTokenizer
        attention_mask = (patches_final_sum != 0).long()

        ones = torch.ones(1, 2)

        attention_mask = torch.cat(
            (ones, attention_mask),
            dim=1,
        )

        return attention_mask


# %%
