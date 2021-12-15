import torch
import shutil
import json
import os
import numpy as np
import logging
from contextlib import contextmanager
from timeit import default_timer
import yaml
import argparse





def save_checkpoint(state,checkpoint_dir):

    filename = os.path.join(checkpoint_dir + '_model_best.pth.tar')
    torch.save(state, filename)
    print("Saving Model in ", filename)
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




def model_selector(config, summary_writer):
    return CLASS_MAPPING[config.model_name](config, summary_writer)


def get_shuffle_buffer_size(dataset, is_training=True):
    if is_training:
        if dataset == 'BEN':
            return 10000  # 39000
        elif dataset == 'DLRSD':
            return 1680
    else:
        return 0


def save_config(config, training_time):
    exp_ids = []
    configs_path = config.dumps.configs
    for config_f in os.listdir(configs_path):
        if '.json' in config_f:
            with open(os.path.join(configs_path, config_f), 'r') as fp:
                contents = json.load(fp)
            exp_ids.append(contents['exp_id'])

    if len(exp_ids) == 0:
        config = config._replace(exp_id=0)
    elif len(np.where(np.array(exp_ids) == config.exp_id)[0]) > 0:
        config = config._replace(exp_id=int(max(exp_ids) + 1))

    config = config._replace(training_time='{:0.1f}'.format(training_time))
    save_file_name = os.path.join(
        configs_path, config.suffix + '.json')

    with open(save_file_name, 'w') as fp:
        res = dict(config._asdict())
        res['dumps'] = dict(config.dumps._asdict())
        json.dump(res, fp)
    return save_file_name


def select_gpu(gpu_number):
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu_number], 'GPU')
        except RuntimeError as e:
            # Visible devices must be set at program startup
            print(e)


@contextmanager
def timer_calc():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end - start


def prep_logger(log_file):
    _FMT_STRING = '[%(levelname)s:%(asctime)s] %(message)s'
    _DATE_FMT = '%Y-%m-%d %H:%M:%S'

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(_FMT_STRING, datefmt=_DATE_FMT))
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(_FMT_STRING, datefmt=_DATE_FMT))
    logger.addHandler(file_handler)


def get_logger():
    """
    Returns the default logger for this project.

    Returns
        logging.Logger: The default logger for this project.
    """

    return logging.getLogger()


def check_h5_metric(file):
    print(file)

    hf = h5py.File(file, 'r')
    keys= list(hf.keys())

    for key in keys:
        out = np.array(hf[key])
        print("{} : {}".format(key,out))


    if "average_precision" in keys:
        for i in [8, 16, 32, 64, 128, 1000]:
            print('mAP@{}(%) {}'.format(i, hf["average_precision"][i - 1] * 100))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Checking h5 metrics')
    parser.add_argument('--filepath', metavar='PATH', help='path to the saved retrieval h5.file')

    args = parser.parse_args()
    check_h5_metric(args.filepath)




