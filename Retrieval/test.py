
import math

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import datetime
from tensorboardX import SummaryWriter
import os
import pickle
from operator import itemgetter


# local imports


##local imports


from utils import get_fusion
from utils import parse_config
from utils import MetricTracker
from utils import save_params
from utils import get_scheduler
from utils import get_optimizer
from utils import save_checkpoint
from utils import get_metrics

from models import get_model
from data import dataGenBigEarthLMDB_joint
from loss import get_loss_func


def test(filename,archive_path):
    save_path = os.path.join(config['logging_params']['save_dir'], config['name'],
                             config['logging_params']['name'])
    print('saving file name is ', save_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using {} device".format(device))

    config = parse_config(filename)
    ### data generation data loader preperation
    train_dataGen = dataGenBigEarthLMDB_joint(
        bigEarthPthLMDB_S2=config["bigEarthPthLMDB_S2"],
        bigEarthPthLMDB_S1=config["bigEarthPthLMDB_S1"],
        state='train',
        train_csv=config["train_csv"],
        val_csv=config["val_csv"],
        test_csv=config["test_csv"]
    )

    query_data_loader = DataLoader(train_dataGen, config["batch_size"], num_workers=0, shuffle=True, pin_memory=True)
    ### data generation data loader preperation

    model = get_model(config["type"], config["n_features"], config["projection_dim"], config["out_channels"])
    model.to(device)

    loss_func = get_loss_func(config["loss_func"], config["projection_dim"], config["fusion"])
    loss_func.to(device)


    # feature_dict: dict - contains for each key a tuple of
    # (torch.tensor: projection_head_s1,torch.tensor: projection_head_s2, fusion, label of archive_image)
    feature_dict = pickle.load(open(archive_path, "rb"))

    for epoch in range(config["start_epoch"], config["epochs"]):
        print('Epoch {}/{}'.format(epoch + 1, config["epochs"]))
        print('-' * 10)
        retrieve_CM(query_modality, feature_dict, model, query_data_loader, test_writer,epoch, config, device)


    pass

def retrieve_MM( model, query_loader,  epoch,  config, device):
    model.eval()

    loss_tracker = MetricTracker()

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(retrieve_loader, desc="retrieve")):
            imgs_S1 = batch["bands_S1"].to(device)
            imgs_S2 = batch["bands_S2"].to(device)

            labels = batch['labels'].to(device)

            h_i, h_j, projection_i, projection_j = model(imgs_S1, imgs_S2)
            # projection_i and j are the outputs after the mlp heads

            fused = get_fusion(config["fusion"], projection_i, projection_j)
    pass

def retrieve_CM( query_modality,feature_dict,num_retrieved,model, query_loader,  test_writer,epoch,  config, device):
    """
    :param query_modality: string - describes the modality S1 or S2 of the query
    :param feature_dict: feature dictionary containing feature_dict: dict - contains for each key a tuple of(torch.tensor: projection_head_s1,torch.tensor: projection_head_s2, fusion, label of archive_image)
    :param model: neural network
    :param query_loader: DataLoader to load one query at the time, !set batch_size to 1!
    :param epoch: epoch
    :param retrieve_writer: retrieve writer
    :param config: configs with parameters for training etc., conf["fusion"] has to be considered, same as in train ?
    :param device: compute device


    """
    model.eval()

    precision_tracker = MetricTracker()
    recall_tracker = MetricTracker()

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(query_loader, desc="retrieve")):

            if query_modality == "S1" :
                query = batch["bands_S1"].to(device)
            elif query_modality == "S2" :
                query = batch["bands_S2"].to(device)

            labels_query = batch['labels'].to(device)

            all_fused_queries= calculate_fused_query(query, model, feature_dict, query_modality, config, device)

            retrieved_labels = sim_score(all_fused_queries,feature_dict)

            metrics_dict = get_metrics(labels_query, retrieved_labels,num_retrieved)

            Precision = "Total_Precison@{}"
            Recall = "Total_Recall@{}"

            precision_tracker.update(metrics_dict[Precision.format(num_retrieved)])
            recall_tracker.update(metrics_dict[Precision.format(num_retrieved)])

    info = {
        "Recall": recall_tracker,
        "Precision":precision_tracker,
    }

    for tag, value in info.items():
        test_writer.add_scalar(tag, value, epoch)

    print('Precision: {:.6f}, Recall: {:.6f}'.format(
        precision_tracker,recall_tracker.avg)
    )

def calculate_fused_query(query,model,feature_dict,query_modality,config,device):
    """

    :param query: torch.tensor
    :param model: neural network
    :param feature_dict: feature dictionary containing feature_dict: dict - contains for each key a tuple of(torch.tensor: projection_head_s1,torch.tensor: projection_head_s2, fusion, label of archive_image)
    :param query_modality: string - describes the modality S1 or S2 of the query
    :param config: configs with parameters for training etc., conf["fusion"] has to be considered, same as in train ?
    :return:

    all_fused_queries: list containing tuples (fused_query,label_archive_idx) of query and the ith image/label (X_a_i^crossmodality) of the archive

    """
    all_fused_queries = []


    ## for each cross modality in feature dict calculate fusion and similarity score
    for idx in range(len(feature_dict)):

        if query_modality == "S1":
            imgs_S2 = torch.squeeze(feature_dict[idx][1]).tolist()  # dim = 1 contains images from Modality S2
            h_i, h_j, projection_i, projection_j = model(query, imgs_S2.to(device))


        elif query_modality == "S2":
            imgs_S1 = torch.squeeze(feature_dict[idx][0]).tolist()  # dim = 0 contains images from Modality S1
            h_i, h_j, projection_i, projection_j = model(imgs_S1.to(device), query)

        # projection_i and j are the outputs after the mlp heads

        fused_query = get_fusion(config["fusion"], projection_i, projection_j)

        all_fused_queries.append((fused_query,feature_dict[idx][3]))

    return all_fused_queries

def sim_score(all_fused_queries,feature_dict):
    """
    Returns similarity score for all fusion queries  and all fusioon_archive vectors

    :param all_fused_queries: list containing tuples (fused_query,label_archive_idx) of query and the ith image/label (X_a_i^crossmodality) of the archive

    :param feature_dict: feature dictionary containing feature_dict: dict - contains for each key a tuple of(torch.tensor: projection_head_s1,torch.tensor: projection_head_s2, fusion, label of archive_image)

    :return:

    tuple-    labels sorted from max to min sim_scores are returned as torch.tensors[1,19]

    """
    assert len(all_fused_queries)== len(feature_dict)
    pairwise_distance = lambda u, v: 0.5 * np.sum(((u - v) ** 2) / (u + v + 1e-10))
    similarities = []

    for idx in range(len(feature_dict)):


        sim = pairwise_distance(all_fused_queries[idx][0].cpu().detach().numpy(),feature_dict[idx][2].cpu().detach().numpy())

        similarities.append((sim,all_fused_queries[idx][1]))

    similarities = sorted(similarities,reverse=True)
    #sim_scores were sorted from max to min as tuples with associated label_archive_idx description (ith image/label (X_a_i^crossmodality) of the archive)

    return tuple(map(itemgetter(1), similarities))

