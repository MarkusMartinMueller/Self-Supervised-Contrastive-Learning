import torch

def get_optimizer(model,optimizer_name,LR,weight_decay):
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                lr=LR,
                weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                lr=LR,
                weight_decay=weight_decay)

    else:
        raise ValueError('Invalid optimizer.')

    return optimizer
