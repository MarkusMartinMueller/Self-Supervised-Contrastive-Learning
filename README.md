# BigEarthNet Multi-Modal Self-Supervised Contrastive CBIR

This repository contains code to use deep learning models, pre-trained on the BigEarthNet archive with PyTorch, to train new models, and to evaluate pre-trained models. 

For BigEarthNet-19 labels, it is highly recommended to first check the [BigEarthNet-19 Deep Learning Models repository](https://git.tu-berlin.de/rsim/BigEarthNet-S2_19-classes_models).


# Training

The script `project/train.py` is for training the CNN models. This file expects the following parameters:

* `--filepath`: Yaml config path to train model

To specify your training and login parameters yaml config files are used. A config file is stored as `args.yaml` in `project/config` and contains following parameters : 

* `batch_size`: Batch size used during training, e.g. 256
* `start_epoch`: 0
* `epochs`: The number of epochs for the training, e.g. 100
* `type`: "joint" # in the supervised setting, decision between joint or separate model
* `train_csv`: The path to the csv file of train patches, "/media/storagecube/markus/splits_mm_serbia/train.csv"
* `val_csv`: The path to the csv file of test patches, "/media/storagecube/markus/splits_mm_serbia/val.csv"
* `test_csv`: The path to the csv file of train patches, "/media/storagecube/markus/splits_mm_serbia/test.csv"
* `loss_func`: The name of the loss function used for training, choices = "classification", "contrastive"
* `fusion`: fusion technique used to combine projection head outputs, choices = "concat", "avg", "max" ,"sum"
* `bigEarthPthLMDB_S2`: LMDB file path previously created for the BigEarthNet Sentinel 2,e.g. "/media/storagecube/markus/project/data/BigEarth_Serbia_Summer_S2.lmdb"
* `bigEarthPthLMDB_S1`:LMDB file path previously created for the BigEarthNet Sentinel 1,e.g. "/media/storagecube/markus/project/data/BigEarth_Serbia_Summer_S1.lmdb"
* `num_cpu`: number of cpus used for the retrieval via ray, e.g. 10

* `name`: sub directory of logs which contains experiments, e.g. "test"
* `n_features`: 2048  # features after the resnet 50 layer, for resnet50 <strong>!unmutable!</strong>
* `projection_dim`: projection dimension for the projection heads, e.g. 128
* `out_channels`: is only used in joint mode for the Resnet model out-channels coming from Conv1,e.g. 32


* `optimizer`: optimizer used for training, choices = "sgd" , "adam"
* `learning_rate`:The initial learning rate
* `weight_decay`: The initial weight decay
* `scheduler`: One Cyle Learning rate scheduler used for sgd optimizer
* `temperature`: The initial temperature parameter for contrastive loss

* `logging_params`:
  * `save_dir`: "logs/" , directory where all loggin parameters are saved
  * `name`: name of the folder where the experiments is saved, e.g. "joint_concat_sgd" 


* `state_dict`: state_directory which contains `checkpoint_##.tar`, e.g. "/media/storagecube/markus/project/logs/test/joint_concat_sgd/checkpoints_model_best.pth.tar"  

<strong>Important note:</strong> 

If you want to run training with different parameters , you always have to change `logging_params/name` and `state_dict` accordingly.<p>
E.g. In `logging_params/name` <strong> joint_concat_sgd</strong> is changed to <strong>joint_concat_adam</strong> then `state_dict` has to be changed to "/media/storagecube/markus/project/logs/test/<strong>joint_concat_adam</strong>/checkpoints_model_best.pth.tar"  </p>


# Retrieval


The script `project/retrieval.py` is for retrieval task . This file expects the following parameters:

* `--filepath` : path to the saved parameters.yaml file, e.g. /media/storagecube/markus/project/logs/test/joint_concat_sgd/parameters.yaml

The script `project/multi_train_retrieval.py` is for training and retrieval task for multiple fusion and model tye parameters . This file expects the following parameters:
* `folder_name`: directory where the model checkpoints are saved, which is a sub-directory of logs, e.g. "test"
* `loss_func`: Loss Function for the training

`multi_train_retrieval.py` creates config yaml files on the fly and saves them in `config/` directory. <p>
In the class Config() are <strong>Default values </strong>, which can be changed before running the file.



# Metrics 

The script `utils/utils.py` is for checking the metrics . This file expects the following parameters:
* `--filepath` : path to the saved parameters.yaml file, e.g. /media/storagecube/markus/project/logs/test/joint_concat_sgd/parameters.yaml
At the moment precision, recall (for both the first 10 values) and map are shown. To change that only does are shown you have to  comment line 189: 

        if key == "precision" or key == "recall":

        or change line 190:

        out = np.array(hf[key])[:10]  # change 10 to desired number up to 1000

