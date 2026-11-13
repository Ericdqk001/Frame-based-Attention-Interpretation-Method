# An interpretable speech foundation model for depression detection

Code for the INTERSPEECH 2025 paper:

> Qingkun Deng, Saturnino Luz, Sofia de la Fuente Garcia. *An interpretable speech
> foundation model for depression detection by revealing prediction-relevant acoustic
> features from long speech.* [arXiv:2406.03138](https://arxiv.org/abs/2406.03138)

The repository contains the speech-level Audio Spectrogram Transformer (AST), the
segment-level AST baseline, and the frame-based attention interpretation method.

## Repository structure

```
scripts/adapt_params.py        Convert the SSAST checkpoint into the model's parameter format
src/train.py                   Entry point for training the speech-level model
src/train/utils.py             Training loop for the speech-level model
src/train/baseline_utils.py    Training loop for the segment-level baseline
src/model/                     Model definitions (frame-based AST, speech-level encoder)
src/load/load.py               Datasets
src/interpret/                 Interpretation method (attention relevance maps)
```

## Installation

Python 3.10 or 3.11. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions used for the paper (PyTorch 2.0.1, Transformers 4.36.2).

## Pretrained weights

The sentence-level encoder is initialised from **SSAST-Base-Frame-400**
([Gong et al., 2022](https://github.com/YuanGongND/ssast)), a frame-based AST
pretrained with self-supervision on AudioSet and LibriSpeech.

1. Download `SSAST-Base-Frame-400.pth` from the
   [SSAST repository](https://github.com/YuanGongND/ssast) (listed under its pretrained
   models).
2. Convert it to the parameter names used by this code:

   ```bash
   python scripts/adapt_params.py --ssast-checkpoint path/to/SSAST-Base-Frame-400.pth
   ```

   This writes `data/pretrained_params/adapted_frame_based_AST_params.pt`, which training
   loads by default.

The conversion renames the parameters to the Hugging Face AST convention, splits the
fused query/key/value projections, and drops the SSAST pretraining heads. It also sets the
**positional embeddings to zero**: the overlapping frame patches used here (128 × 2,
time stride 1) produce 1025 tokens, which does not match the SSAST positional embeddings.
The embeddings are frozen during training, so the trained model has no positional
embeddings in the sentence-level encoder.

## Data

The speech data used in the paper is not publicly available. To train on your own data,
provide the following, relative to the repository root:

```
annotations/train_speech_level.csv     columns: id, label (0 or 1)
annotations/test_speech_level.csv
data/speech_level_train/<id>/<id>-<start>-<end>.wav
data/speech_level_test/<id>/<id>-<start>-<end>.wav
```

Each participant has a folder of recordings, one per sentence, named so that `<start>`
gives the start time used to order them. Up to 42 recordings per participant are used.
The paths can be changed in `build_loader()` in `src/train/utils.py`.

## Training

From the repository root:

```bash
python src/train.py
```

The hyperparameters in `src/train.py` are those used for the paper:

| Setting | Value |
|---|---|
| Epochs | 10 |
| Learning rate | 3e-5, linear warmup over the first 10% of steps, then linear decay |
| Gradient accumulation | 64 participants per update (batch size 1) |
| Weight decay | 0.01 |
| Speech-level encoder layers | 6 |
| Sentence-level AST | Embeddings and layers 0–8 frozen; last 3 layers reinitialised and trained |

`train_sample_num=555` is the number of training participants in the paper's dataset.
It is only used to compute the learning-rate schedule; set it to the size of your own
training set when training on different data.

After each epoch the script reports the training loss and the loss and accuracy on the
test split; this monitoring does not affect training. Checkpoints are saved for epochs
6–10 as `model_weights<epoch>.pt` in the working directory. The model reported in the
paper is the epoch 9 checkpoint. Training took about 4 hours on one NVIDIA A100 GPU.

To load a trained checkpoint without the SSAST file:

```python
import torch
from model.SpeechLevelAST import SpeechLevelAST  # with src/ on the Python path

model = SpeechLevelAST(
    reinit_encoders_n_layers=3,
    num_layers_pl_encoder=6,
    params_file_path=None,
)
model.load_state_dict(torch.load("model_weights9.pt", map_location="cpu"))
```

### Segment-level baseline

`src/train/baseline_utils.py` trains the segment-level AST baseline. It has no entry
script; call `train(config)` with a config that provides `epochs`, `batch_size`,
`lr_ast`, `reinit_encoders_n_layers`, `train_sample_num` and `weight_decay`. It reads
`annotations/{train,test}_segment_level_42.csv` (columns `id`, `file_stem`, `label`) and
`data/segment_level_{train,test}/<file_stem>.wav`.

## Interpretation

The relevance method is implemented in `src/interpret/scripts/relevancy_generator.py`.
It combines attention maps and their gradients across layers, following
[Chefer et al. (2021)](https://github.com/hila-chefer/Transformer-Explainability), to
score each recording (speech level) and each spectrogram frame (sentence level).
`src/interpret/scripts/highlight_spec.py` maps frame relevance back onto the spectrogram.

## Citation

```bibtex
@inproceedings{deng2025interpretable,
  title     = {An interpretable speech foundation model for depression detection by revealing prediction-relevant acoustic features from long speech},
  author    = {Deng, Qingkun and Luz, Saturnino and de la Fuente Garcia, Sofia},
  booktitle = {Proc. Interspeech 2025},
  year      = {2025}
}
```

Please also cite SSAST (Gong et al., 2022), whose pretrained model this work builds on.

## License

MIT. See [LICENSE](LICENSE).
