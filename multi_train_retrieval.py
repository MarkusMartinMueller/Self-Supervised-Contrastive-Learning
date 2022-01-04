import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import yaml

from utils import parse_config, prep_logger, get_logger, timer_calc
from train import main
from retrieval import Retrieval


class Config():
    """Helper class to build config object"""

    def __new__(cls, contents):
        DEFAULTS = {
            "batch_size": 256,
            "start_epoch": 0,
            "epochs": 50,
            "type": "",  # in the supervised setting, decision between joint or separate model
            "train_csv": "/media/storagecube/markus/splits_mm_serbia/train.csv",
            "val_csv": "/media/storagecube/markus/splits_mm_serbia/val.csv",
            'test_csv': "/media/storagecube/markus/splits_mm_serbia/test.csv",
            'loss_func': "classification",
            'fusion': "",
            'bigEarthPthLMDB_S2': "/media/storagecube/markus/project/data/BigEarth_Serbia_Summer_S2.lmdb",
            'bigEarthPthLMDB_S1': "/media/storagecube/markus/project/data/BigEarth_Serbia_Summer_S1.lmdb",
            "num_cpu": 10,

            # model options
            'name': "test",
            'n_features': 2048,  # features after the resnet 50 layer
            'projection_dim': 128,  # "[...] to project the representation to a n-dimensional latent space"
            'out_channels': 32,  # only used in joint mode for the Resnet model out-channels coming from Conv1

            # loss options
            'optimizer': "adam",
            'learning_rate': 0.0001,
            'weight_decay': 0.0,
            'scheduler': False,  # optional not used right now
            'temperature': 0,  # for NXT Loss

            ## logging options

            'logging_params': [{
                'save_dir': "logs/",
                'name': "",  # maybe a special name
            }],

            # reload options
            'state_dict': "",  # set to the directory containing `checkpoint_##.tar`

        }

        for key in list(contents):
            if DEFAULTS[key]:
                DEFAULTS[key] = contents[key]

        configs = list(DEFAULTS)

        ## change file path on erde
        filepath = r"/media/storagecube/markus/project/config/args_{}_{}.yaml".format(contents['type'],
                                                                                      contents['fusion'])
        with open(filepath, 'w') as file:  # create a new yaml file
            data = yaml.dump(DEFAULTS, file)

        return filepath


def train_retrieval(config_dict):
    config = Config(config_dict)
    main(config)

    prep_logger('retrieval.log')
    logger = get_logger()

    with timer_calc() as elapsed_time:
        configs = parse_config(config)

        retrieval = Retrieval(configs)

        retrieval.feature_extraction()
        retrieval.retrieval()
        retrieval.prep_metrics()
        retrieval.finish_retrieval()
        del retrieval
        del configs
        logger.info('Args.yaml is finished within {:0.2f} seconds'.format(elapsed_time()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-Retrieval')

    parser.add_argument('--loss_func', metavar='PATH', help='Loss function used for  training',
                        choices=['classification', 'contrastive'])

    args = parser.parse_args()



    for typ in ['joint', 'separate']:
        for fusion in ['avg', 'max', 'sum', 'concat']:
            config_dict = {
            "loss_func": args.loss_func,
            "type": typ,
            "fusion": fusion,
            "state_dict": "/media/storagecube/markus/project/logs/test/{}_{}_adam/checkpoints_model_best.pth.tar".format(typ, fusion),

            'logging_params': {'save_dir': "logs/",
                               'name': '{}_{}_adam_{}'.format(typ, fusion,args.loss_func),
                               }
            }
            train_retrieval(config_dict)



