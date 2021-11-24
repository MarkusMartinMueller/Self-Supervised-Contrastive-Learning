import torch

def get_scheduler(optimizer,scheduler_gamma):
    if scheduler_gamma:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                gamma=scheduler_gamma)
    else:
        scheduler = None
    return scheduler
