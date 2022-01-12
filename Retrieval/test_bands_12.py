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


    test_writer = SummaryWriter(os.path.join(save_path, 'test'))

    ## tensorboard preparations


    ### data generation data loader preperation. you have to set alternative in dict_concat to True !!!!!!!!!!!!!! that the pipeline works
    test_dataGen = dataGenBigEarthLMDB_joint(
        bigEarthPthLMDB_S2=config["bigEarthPthLMDB_S2"],
        bigEarthPthLMDB_S1=config["bigEarthPthLMDB_S1"],
        state='test',
        train_csv=config["train_csv"],
        val_csv=config["val_csv"],
        test_csv=config["test_csv"]
    )


    test_loader = DataLoader(test_dataGen, config["batch_size"], num_workers=0, shuffle=True, pin_memory=True)

    model = ResNet50_bands_12()
    model.to(device)

    checkpoint = torch.load(config["state_dict"])
    model.load_state_dict(checkpoint["state_dict"])
    print("=> loaded checkpoint from (epoch {})".format( checkpoint['epoch']))

    loss_func = get_loss_func(config["loss_func"], device, config["projection_dim"], config["fusion"],
                              config["temperature"])
    loss_func.to(device)

    test(test_loader,model,save_path,loss_func,config,device)

def test(test_loader, model,save_path ,loss_func,  config, device):
    model.eval()

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

    loss_tracker = MetricTracker()

    y_true = []
    predicted_probs = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="test")):
            imgs = batch["bands"].to(device)


            labels = batch['labels'].to(device)

            logits = model(imgs)


            loss = loss_func(logits, labels)

            if config["fusion"] == "concat":
                fc = torch.nn.Linear(2 * config['projection_dim'], 19).to(device)
            else:
                fc = torch.nn.Linear(config['projection_dim'], 19).to(device)

            probs = torch.sigmoid(fc(logits)).cpu().detach().numpy()

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


        info = {
        'Test loss': loss_tracker.avg,
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

    print("saving metrics...")
    np.save(save_path + '_metrics.npy', info)





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Yaml config path to train model')
    parser.add_argument('--filepath', metavar='PATH', help='path to the saved args.yaml')

    args = parser.parse_args()
    main(args.filepath)
