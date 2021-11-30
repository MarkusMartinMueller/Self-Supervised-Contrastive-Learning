import torch
import shutil

import os
import json
import yaml

def save_checkpoint(state,checkpoint_dir):

    filename = os.path.join(checkpoint_dir + '_model_best.pth.tar')
    torch.save(state, filename)
    print("Saved PyTorch Model State to model.pth")



class MetricTracker(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def parse_config(filename):
    """
    Loads the specified configuration file and returns it as a dictionary.

    Args:
        filename (str): the fully qualified name and path of the configuration file.

    Returns:
        dict[str -> obj]: the parsed configuration
    """

    if os.path.exists(filename):
        try:
            if filename.endswith('.yaml'):
                with open(filename, 'r') as file_handle:
                    return yaml.load(file_handle, Loader=yaml.FullLoader)
            elif filename.endswith('.json'):
                with open(filename, 'r') as file_handle:
                    return json.load(file_handle)
            else:
                raise ValueError(
                    f'The type of the configuration file (YAML or JSON) could not be determined from the extension ("{filename}").')

        except IOError as error:
            raise ValueError(
                f'The config file could not be loaded. Original exception is: {error}')

    else:
        raise ValueError(
            f'The specified config file "{filename}" does not exist.')



def save_params(config):
    save_path = os.path.join(config['logging_params']['save_dir'],config['name'],
                             config['logging_params']['name']
)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file = open(os.path.join(save_path, "parameters.yaml"), "w")
    yaml.dump(config, file)
    file.close()




if __name__ == "__main__":

    print(parse_config("C:/Users/Markus/Desktop/project/config/args.yaml"))