from types import SimpleNamespace

from train.utils import train

# Hyperparameters used for the speech-level model reported in the paper
config = SimpleNamespace(
    epochs=10,
    reinit_encoders_n_layers=3,
    num_layers_pl_encoder=6,
    lr_ast=3e-5,
    accumulation_steps=64,
    # Number of training participants in the paper's dataset. Only used to
    # compute the learning-rate schedule; set it to the size of your own
    # training set when training on different data.
    train_sample_num=555,
    weight_decay=0.01,
)

if __name__ == "__main__":
    train(config)
