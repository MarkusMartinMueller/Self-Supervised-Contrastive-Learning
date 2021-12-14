import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import yaml

from utils import save_config, select_gpu, timer_calc
import json
import traceback
import numpy as np
import multiprocessing
import time



from collections import namedtuple
import json
from datetime import datetime


from utils import get_fusion, parse_config, prep_logger, get_logger, timer_calc, get_shuffle_buffer_size
from models import get_model
from data import dataGenBigEarthLMDB_joint
from train import main
from retrieval import Retrieval


class Config():
    """Helper class to build config object"""

    def __new__(cls, contents):
        DEFAULTS = {
        "batch_size": 256,
        "start_epoch": 0,
        "epochs": 100,
        "pretrain": False,
        "type": "joint", # in the supervised setting, decision between joint or separate model
        "train_csv": "/media/storagecube/markus/splits_mm_serbia/train.csv",
        "val_csv": "/media/storagecube/markus/splits_mm_serbia/val.csv",
        'test_csv': "/media/storagecube/markus/splits_mm_serbia/test.csv",
        'loss_func': "Contrastive",
        'fusion': "concat",
        'bigEarthPthLMDB_S2': "/media/storagecube/markus/project/data/BigEarth_Serbia_Summer_S2.lmdb",
        'bigEarthPthLMDB_S1': "/media/storagecube/markus/project/data/BigEarth_Serbia_Summer_S1.lmdb",

        # model options
        'name': "Resnet50",
        'n_features': 2048,  # features after the resnet 50 layer
        'projection_dim': 128, # "[...] to project the representation to a n-dimensional latent space"
        'out_channels': 32, # only used in joint mode for the Resnet model out-channels coming from Conv1

        # loss options
        'optimizer': "adam",
        'learning_rate': 0.001,
        'weight_decay': 0.0,
        'scheduler_gamma': 0.9, # optional not used right now
        'temperature': 0.1 ,# for NXT Loss



        # reload options
        'state_dict': "", # set to the directory containing `checkpoint_##.tar`
        'epoch_num': 100, # set to checkpoint number
        'reload': False,


        ## logging options

        'logging_params':[{
          'save_dir': "logs/",
          'name': "separate_concat_adam_contrastive", # maybe a special name
          'summaries': "summaries/",
          'suffix': ""}]
        }


        for key in list(contents):
            if DEFAULTS[key]:
                DEFAULTS[key] = contents[key]

        configs = list(DEFAULTS)



        ## change file path on erde
        filepath = r"/media/storagecube/markus/project/config/args_{}_{}_{}.yaml".format(contents['type'],contents['fusion'],contents["loss_func"])
        with open(filepath, 'w') as file:  # create a new yaml file
            data = yaml.dump(DEFAULTS, file)




        return filepath




def train_retrieval(config_dict):
        config = Config(config_dict)
        main(config)


        prep_logger('retrieval.log')
        logger = get_logger()

        with timer_calc() as elapsed_time:
            config = parse_config(config)

            retrieval = Retrieval(config)

            retrieval.feature_extraction()
            retrieval.retrieval()
            retrieval.prep_metrics()
            retrieval.finish_retrieval()
            del retrieval
            del config
            logger.info('Args.yaml is finished within {:0.2f} seconds'.format(elapsed_time()))


if __name__ == "__main__":

    for type in ['separate','joint']:
        for fusion in ['concat','avg','max','sum']:
            config_dict = {
                "type": type,
                "fusion": fusion,
                "state_dict": "/media/storagecube/markus/project/logs/Resnet50/{}_{}_contrastive_adam/checkpoints_model_best.pth.tar".format(type,fusion),
                'logging_params': {'save_dir': "logs/",
                                   'name': '{}_{}'.format(type,fusion),
                                   'summaries': "summaries/",
                                   "suffix": ""}
            }
            train_retrieval(config_dict)



