from types import SimpleNamespace

from train.baseline_utils import train

# Hyperparameters used for the segment-level baseline reported in the paper
config = SimpleNamespace(
    batch_size=128,
    epochs=15,
    reinit_encoders_n_layers=3,
    lr_ast=3e-5,
    # Number of training segments in the paper's dataset. Only used to compute
    # the learning-rate schedule; set it to the size of your own training set
    # when training on different data.
    train_sample_num=20660,
    weight_decay=0.01,
)

if __name__ == "__main__":
    train(config)
