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

def save_checkpoint(state, is_best, epoch, checkpoint):
    if is_best:
        for name in os.listdir(checkpoint):
            if ('best' in name):
                to_remove = os.path.join(checkpoint, name)
                os.remove(to_remove)
        filename = os.path.join(checkpoint, 'epoch_{}__best.pth'.format(epoch))
    else:
        filename = os.path.join(checkpoint, 'epoch_{}_checkpoint.pth'.format(epoch))
    
    torch.save(state, filename)

def save_model_architecture(model, model_path):
    with open(model_path, 'w') as f:
        f.write(str(model))

def record_log(log_file, content):
    with open(log_file, "a") as f:
        f.write(content)