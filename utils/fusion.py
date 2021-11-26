import torch
import numpy as np


def fusion_concat(z_i: torch.tensor,z_j:torch.tensor) -> torch.tensor:
    """

    :param z_i: torch.tensor output from the mpl projection head, shape should be [batch_size,projection_dim]
    :param z_j: torch.tensor output from the mpl projection head, shape should be [batch_size,projection_dim]

    :return:
    concat_fusion: torch.tensor as concatenation along first dim, shape should be [batch_size,2*projection_dim]
    """

    assert z_i.shape == z_j.shape

    return torch.cat((z_i, z_j), 1)



def fusion_avg(z_i: torch.tensor,z_j:torch.tensor) -> torch.tensor:
    """

    :param z_i: torch.tensor output from the mpl projection head, shape should be [batch_size,projection_dim]
    :param z_j: torch.tensor output from the mpl projection head, shape should be [batch_size,projection_dim]

    :return:
    avg_fusion: average value element-wise, shape should be [batch_size,projection_dim]

    """

    assert z_i.shape == z_j.shape

    return torch.add(z_i, z_j)

def fusion_sum(z_i: torch.tensor,z_j:torch.tensor) -> torch.tensor:
    """

    :param z_i: torch.tensor output from the mpl projection head, shape should be [batch_size,projection_dim]
    :param z_j: torch.tensor output from the mpl projection head, shape should be [batch_size,projection_dim]

    :return:
    sum_fusion: sum over first dimension element-wise, shape should be [batch_size,projection_dim]

    """

    assert z_i.shape == z_j.shape

    return torch.add(z_i, z_j)

def fusion_max(z_i: torch.tensor,z_j:torch.tensor) -> torch.tensor:
    """

    :param z_i: torch.tensor output from the mpl projection head, shape should be [batch_size,projection_dim]
    :param z_j: torch.tensor output from the mpl projection head, shape should be [batch_size,projection_dim]

    :return:
    max_fusion: max value over first dimension element-wise, shape should be [batch_size,projection_dim]

    """

    assert z_i.shape == z_j.shape

    return torch.maximum(z_i, z_j)


def get_fusion(name,z_i,z_j):
    if name == "concat":
        return fusion_concat(z_i,z_j)
    elif name == "avg":
        return fusion_avg(z_i, z_j)
    elif name == "sum":
        return fusion_sum(z_i, z_j)
    elif name == "max":
        return fusion_max(z_i, z_j)
    else:
        raise ValueError('Invalid fusion techniques')


if __name__ == "__main__":


    inputs_s1 = torch.randn((4, 128))
    inputs_s2 = torch.randn((4, 128))

    print(fusion_concat(inputs_s1, inputs_s2).shape)
    print(fusion_avg(inputs_s2, inputs_s1).shape)
    print(fusion_sum(inputs_s2, inputs_s1).shape)
    print(fusion_max(inputs_s2, inputs_s1).shape)
