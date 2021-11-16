import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import datetime
from tensorboardX import SummaryWriter
import os




##local imports

from utils.config import parse_config
from utils.utils import MetricTracker



def train(trainloader, model, criterion , optimizer,  epoch,train_writer):

    loss_tracker = MetricTracker()

    model.train()

    for idx, data in enumerate(tqdm(trainloader, desc="training")):
        #imgs = data['img'].to(torch.device("cuda"))
        #labels = data['label'].to(torch.device("cuda"))

        logits = model(imgs)

        ### detach gradients
        optimizer.zero_grad()

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        loss_tracker.update(loss.item(), imgs.size(0))

    info = {
        "Loss": loss.tracker.avg,
    }

    for tag, value in info.items():
        train_writer.add_scalar(tag, value, epoch)

    print('Train Loss: {:.6f}'.format(
        loss_tracker.avg
    ))




def val(valloader, model, criterion , optimizer,  epoch, val_writer):

    model.eval()

    logits = []
    y_true = []

    with torch.no_grad():
        for idx, data in enumerate(tqdm(valloader, desc="validation")):
            imgs = data['img'].to(torch.device("cuda"))
            labels = data['label'].to(torch.device("cpu"))

            logits_batch = model(imgs)

            y_true += list(labels.numpy())
            logits += list(logits_batch.cpu().numpy())

    y_true = np.asarray(y_true)
    logits = np.asarray(logits)

    y_pred = np.argmax(logits, axis=1)



    info = {
        'Acc': acc
    }
    for tag, value in info.items():
        val_writer.add_scalar(tag, value, epoch)

    print('Test Acc: {:.6f} '.format(
        acc,
    ))

    return acc


def main():


    config = parse_config(filename)

    ## tensorboard preparations
    sv_name = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
    print('saving file name is ', sv_name)

    checkpoint_dir = os.path.join('./', sv_name, 'checkpoints')
    logs_dir = os.path.join('./', sv_name, 'logs')
    train_writer = SummaryWriter(os.path.join(logs_dir, 'runs', sv_name, 'training'))
    val_writer = SummaryWriter(os.path.join(logs_dir, 'runs', sv_name, 'val'))
    ## tensorboard preparations

    train_data_loader = DataLoader(train_data, config["batch_size"], num_workers=4, shuffle=True, pin_memory=True)

    for epoch in range(config["start_epoch"], config["epochs"]):
        print('Epoch {}/{}'.format(epoch, config["epochs"] - 1))
        print('-' * 10)


if __name__ == "__main__":

    main("C:\Users\Markus\Desktop\project\config\args.yaml")