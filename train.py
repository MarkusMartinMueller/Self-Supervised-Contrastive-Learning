import math

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import datetime
from tensorboardX import SummaryWriter
import os


# local imports


##local imports


from utils import get_fusion
from utils import parse_config
from utils import MetricTracker
from utils import save_params
from utils import get_scheduler
from utils import get_optimizer
from utils import save_checkpoint

from models import get_model
from data import dataGenBigEarthLMDB_joint
from loss import get_loss_func
from utils import  Precision_score, Recall_score, F1_score, F2_score, Hamming_loss, Subset_accuracy, \
    Accuracy_score, One_error, Coverage_error, Ranking_loss, LabelAvgPrec_score



def main(filename):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using {} device".format(device))

    config = parse_config(filename)

    ## tensorboard preparations
    save_path = os.path.join(config['logging_params']['save_dir'], config['name'],
                             config['logging_params']['name'])
    print('saving file name is ', save_path)

    checkpoint_dir = os.path.join(save_path, 'checkpoints')
    train_writer = SummaryWriter(os.path.join(save_path, 'training'))
    # val_writer = SummaryWriter(os.path.join(save_path, 'val'))
    ## tensorboard preparations

    ### data generation data loader preperation
    train_dataGen = dataGenBigEarthLMDB_joint(
        bigEarthPthLMDB_S2=config["bigEarthPthLMDB_S2"],
        bigEarthPthLMDB_S1=config["bigEarthPthLMDB_S1"],
        state='train',
        train_csv=config["train_csv"],
        val_csv=config["val_csv"],
        test_csv=config["test_csv"]
    )

    val_dataGen = dataGenBigEarthLMDB_joint(
        bigEarthPthLMDB_S2=config["bigEarthPthLMDB_S2"],
        bigEarthPthLMDB_S1=config["bigEarthPthLMDB_S1"],
        state='val',
        train_csv=config["train_csv"],
        val_csv=config["val_csv"],
        test_csv=config["test_csv"]
    )

    train_data_loader = DataLoader(train_dataGen, config["batch_size"], num_workers=4, shuffle=True, pin_memory=True)
    val_data_loader = DataLoader(val_dataGen, config["batch_size"], num_workers=4, shuffle=True, pin_memory=True)

    ### data generation data loader preperation

    ## model,optimizer, scheduler, loss_func initilization

    model = get_model(config["type"], config["n_features"], config["projection_dim"], config["out_channels"])
    model.to(device)

    optimizer = get_optimizer(model, config["optimizer"], config["learning_rate"], config["weight_decay"])
    scheduler = get_scheduler(optimizer, config["scheduler"], config["learning_rate"], config["epochs"],
                              train_data_loader)
    loss_func = get_loss_func(config["loss_func"], device,config["projection_dim"], config["fusion"],config["temperature"])
    loss_func.to(device)

    ## save params in yaml file
    save_params(config)

    min_val_loss = math.inf

    #pretrained = True
    #if pretrained:
        #print("=> loading checkpoint '{}'".format(torch.load(config["state_dict"])["epoch"]))
        #model.load_state_dict(torch.load(config["state_dict"])["state_dict"])
        #optimizer.load_state_dict(torch.load(config["state_dict"])["optimizer"])
        #

    for epoch in range(config["start_epoch"], config["epochs"]):
        print('Epoch {}/{}'.format(epoch + 1, config["epochs"]))
        print('-' * 10)

        train(model, train_data_loader, loss_func, optimizer, scheduler,epoch, train_writer, config, device)

        if epoch % 2 == 0:

            val_loss = val(val_data_loader, model, loss_func, epoch, train_writer, config, device)

            if val_loss < min_val_loss:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, checkpoint_dir)

                min_val_loss = val_loss
    #train_writer.close()
    # val_writer.close()


def train(model, trainloader, loss_func, optimizer, scheduler,epoch, train_writer, config, device):
    loss_tracker = MetricTracker()

    model.train()

    for idx, batch in enumerate(tqdm(trainloader, desc="training")):
        imgs_S1 = batch["bands_S1"].to(device)
        imgs_S2 = batch["bands_S2"].to(device)

        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        h_i, h_j, projection_i, projection_j = model(imgs_S1,
                                                     imgs_S2)  # projection_i and _j are the outputs after the mlp heads

        fused = get_fusion(config["fusion"], projection_i, projection_j)

        if config["loss_func"] == "classification":
            loss = loss_func(fused, labels)
        elif config["loss_func"] == "contrastive":
            loss = loss_func(projection_i,projection_j)

        ### detach gradients

        loss.backward()
        optimizer.step()
        if config['scheduler']:
            scheduler.step()

        loss_tracker.update(loss.item())

    info = {
        "Loss": loss_tracker.avg,
    }

    for tag, value in info.items():
        train_writer.add_scalar(tag, value, epoch)

    print('Train Loss: {:.6f}'.format(
        loss_tracker.avg
    ))

    # if config['scheduler_gamma']:
    # scheduler.step()


def val(valloader, model, loss_func, epoch, val_writer, config, device):
    prec_score_ = Precision_score()
    recal_score_ = Recall_score()
    f1_score_ = F1_score()
    f2_score_ = F2_score()
    hamming_loss_ = Hamming_loss()
    subset_acc_ = Subset_accuracy()
    acc_score_ = Accuracy_score()
    one_err_ = One_error()
    coverage_err_ = Coverage_error()
    rank_loss_ = Ranking_loss()
    labelAvgPrec_score_ = LabelAvgPrec_score()

    model.eval()

    loss_tracker = MetricTracker()

    y_true = []
    predicted_probs = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(valloader, desc="validation")):
            imgs_S1 = batch["bands_S1"].to(device)
            imgs_S2 = batch["bands_S2"].to(device)

            labels = batch['labels'].to(device)

            h_i, h_j, projection_i, projection_j = model(imgs_S1, imgs_S2)
            # projection_i and j are the outputs after the mlp heads

            fused = get_fusion(config["fusion"], projection_i, projection_j)
            if config["loss_func"] == "classification":
                loss = loss_func(fused, labels)
            elif config["loss_func"] == "contrastive":
                loss = loss_func(projection_i, projection_j)

            probs = torch.sigmoid(loss).cpu().numpy()

            predicted_probs += list(probs)
            y_true += list(labels.cpu().numpy())

            loss_tracker.update(loss.item())

    predicted_probs = np.asarray(predicted_probs)
    y_predicted = (predicted_probs >= 0.5).astype(np.float32)
    y_true = np.asarray(y_true)

    macro_f1, micro_f1, sample_f1 = f1_score_(y_predicted, y_true)
    macro_f2, micro_f2, sample_f2 = f2_score_(y_predicted, y_true)
    macro_prec, micro_prec, sample_prec = prec_score_(y_predicted, y_true)
    macro_rec, micro_rec, sample_rec = recal_score_(y_predicted, y_true)
    hamming_loss = hamming_loss_(y_predicted, y_true)
    subset_acc = subset_acc_(y_predicted, y_true)
    macro_acc, micro_acc, sample_acc = acc_score_(y_predicted, y_true)

    one_error = one_err_(predicted_probs, y_true)
    coverage_error = coverage_err_(predicted_probs, y_true)
    rank_loss = rank_loss_(predicted_probs, y_true)
    labelAvgPrec = labelAvgPrec_score_(predicted_probs, y_true)

    info = {        'Val loss': loss_tracker.avg,
                    "macroPrec": macro_prec,
                    "microPrec": micro_prec,
                    "samplePrec": sample_prec,
                    "macroRec": macro_rec,
                    "microRec": micro_rec,
                    "sampleRec": sample_rec,
                    "macroF1": macro_f1,
                    "microF1": micro_f1,
                    "sampleF1": sample_f1,
                    "macroF2": macro_f2,
                    "microF2": micro_f2,
                    "sampleF2": sample_f2,
                    "HammingLoss": hamming_loss,
                    "subsetAcc": subset_acc,
                    "macroAcc": macro_acc,
                    "microAcc": micro_acc,
                    "sampleAcc": sample_acc,
                    "oneError": one_error,
                    "coverageError": coverage_error,
                    "rankLoss": rank_loss,
                    "labelAvgPrec": labelAvgPrec

                    }
    for tag, value in info.items():
        val_writer.add_scalar(tag, value, epoch)

    print('Validation Loss: {:.6f}'.format(
        loss_tracker.avg
    ))

    return loss


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Yaml config path to train model')
    parser.add_argument('--filepath', metavar='PATH', help='path to the saved args.yaml')

    args = parser.parse_args()
    main(args.filepath)
    #main("C:/Users/Markus/Desktop/project/config/args.yaml")