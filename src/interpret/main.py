from pathlib import Path

import torch
from load.load_local import SpeechLevelDataset
from model.SpeechLevelAST_local import SpeechLevelAST
from torch.utils.data import DataLoader

from interpret.scripts.plot import (
    interpret_sentence_level,
)
from interpret.scripts.relevancy_generator import InterpretationGenerator

checkpoint_path = Path("artifacts/lucky-monkey-16:v0/speech_level_weights.pt")

dataset = SpeechLevelDataset(
    waveform_normalize=False,
    mean=-2.9899,
    std=4.3509,
)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=None,
    shuffle=True,
)

for data in dataloader:
    if data["id"] == "473":
        batch = data
        break

device = "cpu"

model = SpeechLevelAST(
    reinit_encoders_n_layers=2,
    num_layers_pl_encoder=6,
).to(device)

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=torch.device(device),
    )
)

audio_inputs = batch["audio_inputs"].to(device)
audio_attn_masks = batch["audio_attn_masks"].to(device)

print(batch["id"])

generator = InterpretationGenerator(model)

generator(
    audio_inputs=audio_inputs,
    audio_attn_masks=audio_attn_masks,
)

# speech_level_interpretation = interpret_speech_level(
#     generator=generator,
#     num_sentences_indexes=10,
# )

interpret_sentence_level(
    generator,
    sentence_index=31,
    audio_inputs=audio_inputs,
)

# all_specs = []
# all_relevancy_maps = []

# for sentence_index in range(10):  # Assuming 10 sentence indexes
#     # Assuming you have a way to get `audio_inputs` and `reconstructed_relevancy`
#     spec = process_spec(
#         audio_inputs[sentence_index],
#     )
#     relevancy_map = process_relevancy_map(
#         generator,
#         sentence_index,
#         audio_inputs[sentence_index],
#     )

#     all_specs.append(spec)
#     all_relevancy_maps.append(relevancy_map)

# concatenate_and_plot(
#     all_specs,
#     all_relevancy_maps,
#     speech_level_interpretation[:5],
# )
