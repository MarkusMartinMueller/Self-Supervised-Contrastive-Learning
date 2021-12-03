import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms, datasets
import os
import pickle

# local imports
from utils import get_fusion
from utils import parse_config
from models import  get_model
from data import dataGenBigEarthLMDB_joint


def build_db(filename, state_dict_path,state):
    """
    To build the archive at the moment the value of batch_size has to be set to 1, so that
    each image is saved at the time

    :param filename: contains path to parameters.yaml used to train the model
    :param state:  contains string of either val or test to create val or test archive
    """
    # define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Computation device: ', device)

    # initialize the model
    config = parse_config(filename)

    model = get_model(config["type"], config["n_features"], config["projection_dim"], config["out_channels"])
    model.to(device)

    # Load state dict

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(state_dict_path)["state_dict"])

    else:
        model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu'))["state_dict"])

    retrieve_dataGen = dataGenBigEarthLMDB_joint(
        bigEarthPthLMDB_S2 = "C:/Users/Markus/Desktop/project/data/BigEarth_Serbia_Summer_S2.lmdb",
        bigEarthPthLMDB_S1="C:/Users/Markus/Desktop/project/data/BigEarth_Serbia_Summer_S1.lmdb",
        state="val",
        train_csv=config["train_csv"],
        val_csv=config["val_csv"],
        test_csv=config["test_csv"]
    )

    retrieve_data_loader = DataLoader(retrieve_dataGen, config["batch_size"], num_workers=0, shuffle=True, pin_memory=True)

    feature_dict = {}
    model.eval()



    with torch.no_grad():
        for idx, batch in enumerate(tqdm(retrieve_data_loader, desc=state)):
            imgs_S1 = batch["bands_S1"].to(device)
            imgs_S2 = batch["bands_S2"].to(device)

            labels = batch['labels'].to(device)

            h_i, h_j, projection_i, projection_j = model(imgs_S1, imgs_S2)
            # projection_i and _j are the outputs after the mlp heads

            fused = get_fusion(config["fusion"],projection_i,projection_j)



            #feature_dict[idx] = (projection_i, projection_j,fused,labels)
            feature_dict[idx] = (imgs_S1, imgs_S2, fused, labels)
    pickle.dump(feature_dict, open("archive_separate_avg.p", "wb"))


if __name__ == '__main__':
    build_db(filename="C:/Users/Markus/Desktop/project/logs/Resnet50/separate_avg/parameters.yaml", state_dict_path="C:/Users/Markus/Desktop/project/logs/Resnet50/separate_avg/checkpoints_model_best.pth.tar", state="val")
    feature_dict = pickle.load(open("archive_separate_avg.p", "rb"))

    print(feature_dict)


