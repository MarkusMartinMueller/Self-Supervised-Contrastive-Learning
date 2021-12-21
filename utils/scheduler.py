import torch

def get_scheduler(optimizer, scheduler, max_lr, epochs, trainloader):

    if scheduler:
        scheduler_model = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                     steps_per_epoch=len(trainloader))

    else:
        scheduler_model = None
    return  scheduler_model

