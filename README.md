# An interpretable speech foundation model for depression detection

Code for [arXiv:2406.03138](https://arxiv.org/abs/2406.03138): a speech-level Audio
Spectrogram Transformer (AST) for depression detection, and a segment-level AST baseline.

## Setup

Python 3.10 or 3.11, from the repository root:

```bash
pip install -r requirements.txt
```

## Pretrained weights

The model is initialised from **SSAST-Base-Frame-400** ([Gong et al., 2022](https://github.com/YuanGongND/ssast)).
Download `SSAST-Base-Frame-400.pth` from the SSAST repository, then convert it:

```bash
python scripts/adapt_params.py --ssast-checkpoint path/to/SSAST-Base-Frame-400.pth
```

This writes `data/pretrained_params/adapted_frame_based_AST_params.pt`. The conversion
sets the positional embeddings to zero; they stay frozen during training.

## Data

The paper's data is not public. To train on your own data, provide:

```
annotations/{train,test}_speech_level.csv        columns: id, label (0/1)
data/speech_level_{train,test}/<id>/<id>-<start>-<end>.wav
```

Recordings must be mono WAV files (other sample rates are resampled to 16 kHz). One
folder per participant, one recording per sentence, ordered by `<start>`; up to 42
recordings per participant are used.

Spectrograms are normalised as `(x - mean) / std * 0.5`. The mean/std in the code
(`src/load/load.py`, `src/train/baseline_utils.py`) were most likely computed on the
paper's data, but their origin is uncertain. For new data, compute your own or use the
AST AudioSet values: mean -4.2677393, std 4.5689974.

## Training

```bash
python src/train.py
```

`src/train.py` holds the paper's settings (10 epochs in total). `train_sample_num=555` is
the paper's number of training participants and only sets the learning-rate schedule;
change it for your data. Loss on the test split is printed after each epoch, and
checkpoints for epochs 6–10 are saved as `model_weights<epoch>.pt`.

To load a trained checkpoint (with `src/` on the Python path):

```python
model = SpeechLevelAST(reinit_encoders_n_layers=3, num_layers_pl_encoder=6, params_file_path=None)
model.load_state_dict(torch.load("model_weights9.pt", map_location="cpu"))
```

**Baseline:** call `train(config)` in `src/train/baseline_utils.py`. It reads
`annotations/{train,test}_segment_level_42.csv` (`id`, `file_stem`, `label`) and
`data/segment_level_{train,test}/<file_stem>.wav`.

## Citation

```bibtex
@misc{deng2024interpretable,
  title         = {An interpretable speech foundation model for depression detection by revealing prediction-relevant acoustic features from long speech},
  author        = {Deng, Qingkun and Luz, Saturnino and de la Fuente Garcia, Sofia},
  year          = {2024},
  eprint        = {2406.03138},
  archivePrefix = {arXiv}
}
```

Please also cite SSAST (Gong et al., 2022). License: MIT.
