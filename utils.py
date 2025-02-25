import os
import json
import shutil
import torch

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def save_config(config, config_path):
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def save_checkpoint(model_state, optim_state, sche_state, is_best, epoch, checkpoint, value, backbone=False):
    """
    Save model checkpoint
    """
    if backbone:
        back = '_backbone'
    else:
        back = ''
    if is_best:
        for name in os.listdir(checkpoint):
            if ('best{}'.format(back) in name):
                to_remove = os.path.join(checkpoint, name)
                os.remove(to_remove)
        filename = os.path.join(checkpoint, 'epoch_{}_{}_best{}.pth'.format(epoch, value, back))
    else:
        filename = os.path.join(checkpoint, 'epoch_{}_{}_checkpoint{}.pth'.format(epoch, value, back))
    
    torch.save({    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optim_state,
                    'scheduler_state_dict': sche_state,
                    }, filename)

def save_model_architecture(model, model_path):
    with open(model_path, 'w') as f:
        f.write(str(model))

def record_log(log_file, content):
    with open(log_file, "a") as f:
        f.write(content)