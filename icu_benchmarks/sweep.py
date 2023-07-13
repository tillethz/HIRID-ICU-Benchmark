#import wandb
import argparse
import os
from icu_benchmarks.models.train import train_with_gin

from torch.utils.tensorboard import SummaryWriter

def train_config(config, gin_dir, logdir):

    if config["architecture"] == "TCN":
       gin_config = os.path.join(gin_dir, "TCN.gin")
    elif config["architecture"] == "LSTM":
       gin_config = os.path.join(gin_dir, "LSTM.gin")
    else:
       raise NotImplementedError(f"Architecture {config['architecture']} not implemented")
    print(gin_config)
    gin_bindings = get_bindings_and_params(config)
    print(gin_bindings)
    return train_with_gin(model_dir=logdir,
                        overwrite=False,
                        load_weights=False,
                        gin_config_files=[gin_config],
                        gin_bindings=gin_bindings,
                        seed=666, reproducible=True)


def main():
    parser = argparse.ArgumentParser(description='Script to pass the config to the trainer')
    parser.add_argument('--config-path',dest='config_path', help='Path to yaml config', default='config.yaml')
    parser.add_argument('--gin-dir',dest='gin_dir', help='directory of the gin config files', default='configs/hirid/Classification')
    parser.add_argument('--logdir', help='path to the log directory', type=str)
    
    args = parser.parse_args()

    hparam_writer = SummaryWriter(args.logdir)
    config = {  "architecture": "TCN",
                "task": "Dynamic_RespFailure_12Hours",
                "hidden": 64,
                "lr": 3e-4,
                "receptive_field": 2016,
                "dropout": 0.2}
    i = 0
    for dropout in [0.1,0.25]:
        for hidden in [16, 64, 256]:
            for receptive_field in [2016, 288]:
                config["dropout"] = dropout
                config["hidden"] = hidden
                config["receptive_field"] = receptive_field

                log_path = os.path.join(args.logdir, str(i))
                

                metrics = train_config(config, args.gin_dir, log_path)

                hparam_writer.add_hparams(config, metrics, run_name=str(i))
                i+=1



def get_bindings_and_params(config):
    gin_bindings = []

    if "task" in config:
        task = config["task"]
        gin_bindings += ['TASK = "' + str(task)+'"']

    if "hidden" in config:
        hidden = config["hidden"]
        gin_bindings += ['HIDDEN = ' + str(hidden)]

    if "lr" in config:
        lr = config["lr"]
        gin_bindings += ['LR = ' + str(lr)]

    if "receptive_field" in config:
        receptive_field = config["receptive_field"]
        gin_bindings += ['HORIZON = ' + str(receptive_field)]

    if "dropout" in config:
        dropout = config["dropout"]
        gin_bindings += ['DROPOUT = ' + str(dropout)]


    

    return gin_bindings


if __name__ == '__main__':
    main()

#wandb.init(project=wandb_project, entity="failure-prediction", config=config, allow_val_change=True)


#def train():

""" train_with_gin(model_dir=log_dir_seed,
                   overwrite=args.overwrite,
                   load_weights=load_weights,
                   gin_config_files=args.config,
                   gin_bindings=gin_bindings_task,
                   seed=seed, reproducible=reproducible)"""
