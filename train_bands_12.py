import math

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import datetime
from tensorboardX import SummaryWriter
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# local imports


##local imports


from utils import get_fusion
from utils import parse_config
from utils import MetricTracker
from utils import save_params
from utils import get_scheduler
from utils import get_optimizer
from utils import save_checkpoint

from models import get_model, ResNet50_bands_12
from data import dataGenBigEarthLMDB_joint
from loss import get_loss_func


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


    ### data generation data loader preperation. you have to set alternative in dict_concat to True !!!!!!!!!!!!!! that the pipeline works
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

    train_data_loader = DataLoader(train_dataGen, config["batch_size"], num_workers=0, shuffle=True, pin_memory=True)
    val_data_loader = DataLoader(val_dataGen, config["batch_size"], num_workers=0, shuffle=True, pin_memory=True)

    ### data generation data loader preperation

    ## model,optimizer, scheduler, loss_func initilization

    model = ResNet50_bands_12()
    model.to(device)

    optimizer = get_optimizer(model, config["optimizer"], config["learning_rate"], config["weight_decay"])
    # scheduler = get_scheduler(optimizer, config["schedluer_gamma"]
    loss_func = get_loss_func(config["loss_func"], device,config["projection_dim"], config["fusion"],config["temperature"])
    loss_func.to(device)

    ## save params in yaml file
    save_params(config)

    min_val_loss = math.inf

    for epoch in range(config["start_epoch"], config["epochs"]):
        print('Epoch {}/{}'.format(epoch + 1, config["epochs"]))
        print('-' * 10)

        train(model, train_data_loader, loss_func, optimizer, epoch, train_writer, config, device)

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


def train(model, trainloader, loss_func, optimizer, epoch, train_writer, config, device):
    loss_tracker = MetricTracker()

    model.train()

    for idx, batch in enumerate(tqdm(trainloader, desc="training")):
        imgs = batch["bands"].to(device)

        labels = batch['labels'].to(device)

        logits = model(imgs)
        # projection_i and j are the outputs after the mlp heads

        loss = loss_func(logits, labels)

        ### detach gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
    model.eval()

    loss_tracker = MetricTracker()

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(valloader, desc="validation")):
            imgs = batch["bands"].to(device)


            labels = batch['labels'].to(device)

            logits = model(imgs)
            # projection_i and j are the outputs after the mlp heads



            loss = loss_func(logits, labels)


            loss_tracker.update(loss.item())

    info = {
        'Val loss': loss.item(),
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
    # main("C:/Users/Markus/Desktop/project/config/args.yaml")