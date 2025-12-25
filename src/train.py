import wandb
from train.utils import train

wandb.login(key="")

with wandb.init(
    project="Speech-level AST azure test",
    config={
        "epochs": 15,
        "reinit_encoders_n_layers": 3,
        "num_layers_pl_encoder": 6,
        "lr_ast": 3e-5,
        "accumulation_steps": 64,
        "train_sample_num": 555,
        "weight_decay": 0.01,
    },
):
    train(wandb)

# TODO Upload pretrained_params to azure
